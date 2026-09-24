import torch
from fractions import Fraction

from .jazz_features import JazzFeatures


class HarmonyGenerator:
    """Build harmony features and resolve the active chord for each melody onset.

    Chord-progression convention:
        one list element == one bar

    Examples:
        "CMaj7"          -> one chord for the whole bar
        "DMin7 G7"       -> two equally spaced chords in the bar
        "Am7 D7 G7 C7"   -> four equally spaced chords in the bar

    The bar string is only a compact generation-time representation. Before a
    chord is sent to JazzFeatures it is split into an individual chord symbol.
    """

    def __init__(self, max_harmony_len=128, device="cpu"):
        self.max_harmony_len = max_harmony_len
        self.device = device
        self.features = JazzFeatures()

    # ==========================================
    # Chord Normalize
    # Same as training Dataset
    # ==========================================

    def normalize_chord(self, chord):
        if chord is None or chord == "":
            return "N.C"

        chord = chord.strip()

        # Major symbol
        chord = chord.replace("j", "^")
        chord = chord.replace("∆", "^")

        # Minor
        if len(chord) > 1:
            root_end = 1
            if chord[1] in ["b", "#"]:
                root_end = 2
            root = chord[:root_end]
            rest = chord[root_end:]
            if rest.startswith("m"):
                rest = rest.replace("m", "-", 1)
            chord = root + rest

        # Alter chord
        replacements = {
            "alt": "b9#5",
            "ø": "-7b5",
        }

        for old, new in replacements.items():
            chord = chord.replace(old, new)

        return chord

    # ==========================================
    # Bar-spec parsing
    # ==========================================

    def split_bar_chords(self, bar_spec):
        """Return the individual chord symbols contained in one bar string."""
        if bar_spec is None:
            return ["N.C"]

        bar_spec = str(bar_spec).strip()
        if not bar_spec:
            return ["N.C"]

        chords = [part.strip() for part in bar_spec.split() if part.strip()]
        return chords if chords else ["N.C"]

    def flatten_progression(self, bars):
        """Flatten ['CMaj7', 'DMin7 G7'] -> ['CMaj7', 'DMin7', 'G7']."""
        events = []
        for bar_spec in bars:
            events.extend(self.split_bar_chords(bar_spec))
        return events

    # ==========================================
    # Note onset -> active chord
    # ==========================================

    def chord_for_position(
        self,
        bars,
        bar,
        beat,
        division=1,
        tatum=1,
        beats_per_bar=4,
    ):
        """Return the chord active at one melody onset.

        Multiple chords inside a bar are divided evenly across the bar.

        In 4/4:
            "DMin7 G7"
                beat 1-2 -> DMin7
                beat 3-4 -> G7

            "Am7 D7 G7 C7"
                one chord per beat

        DIVISION/TATUM are included so the method also works when a boundary
        falls inside a beat (for example, three equally spaced chords in 4/4).
        """
        if bars is None:
            raise ValueError("bars/chords progression is required")

        bar = int(bar)
        beat = int(beat)
        division = max(int(division), 1)
        tatum = max(1, min(int(tatum), division))
        beats_per_bar = int(beats_per_bar)

        if beats_per_bar < 1:
            raise ValueError("beats_per_bar must be >= 1")
        if bar < 0 or bar >= len(bars):
            raise IndexError(
                f"BAR_{bar} has no chord specification: progression has "
                f"{len(bars)} bars"
            )
        if beat < 1 or beat > beats_per_bar:
            raise ValueError(
                f"BEAT_{beat} is outside a {beats_per_bar}/4 bar"
            )

        bar_chords = self.split_bar_chords(bars[bar])
        n_chords = len(bar_chords)

        if n_chords == 1:
            return bar_chords[0]

        # Exact symbolic onset inside the bar, measured in quarter-note beats.
        onset_in_bar = (
            Fraction(beat - 1, 1)
            + Fraction(tatum - 1, division)
        )

        chord_span = Fraction(beats_per_bar, n_chords)
        chord_index = int(onset_in_bar / chord_span)
        chord_index = max(0, min(chord_index, n_chords - 1))

        return bar_chords[chord_index]

    # Backward-compatible whole-bar lookup.  For a multi-chord bar this returns
    # the first chord only; new generation code should use chord_for_position().
    def chord_for_bar(self, bars, bar):
        return self.chord_for_position(
            bars=bars,
            bar=bar,
            beat=1,
            division=1,
            tatum=1,
            beats_per_bar=4,
        )

    # ==========================================
    # Build Harmony Feature
    # ==========================================

    def build(self, chords, expand_bar_specs=True):
        """Build encoder tensors.

        expand_bar_specs=True is for a bar-level progression where one string
        may contain several chords.  build_note_aligned() disables expansion
        because its input already contains one resolved chord per melody note.
        """
        if expand_bar_specs:
            chords = self.flatten_progression(chords)
        else:
            chords = list(chords)

        input_ids = []
        scale_vectors = []
        chord_tones = []
        guide_vectors = []
        tensions = []
        available_tensions = []
        avoids = []

        Function_Map = {
            "Tonic": 0,
            "SubDominant": 1,
            "Dominant": 2,
        }

        for chord in chords:
            chord = self.normalize_chord(chord)
            feature = self.features.analyze_chord(chord)
            function_id = Function_Map.get(feature["function"], 0)

            input_ids.append(
                [
                    feature["chord_id"],
                    feature["root"],
                    feature["bass"],
                    feature.get("bass_interval", 0),
                    feature.get("inversion", 0),
                    feature["attribute"],
                    function_id,
                    feature.get("level", 0),
                    feature.get("scale", 0),
                    feature.get("duration", 0),
                    feature.get("beat", 0),
                    feature.get("section", 0),
                    feature.get("time", 0),
                ]
            )

            scale_vectors.append(feature["scale_vector"])
            chord_tones.append(feature["chord_tones"])
            guide_vectors.append(feature["guide_tones"])
            tensions.append(feature["tensions"])
            available_tensions.append(feature["available_tensions"])
            avoids.append(feature["avoid"])

        if not input_ids:
            raise RuntimeError("No valid harmony events were built")

        # ==================================
        # Padding
        # ==================================

        length = len(input_ids)
        if length > self.max_harmony_len:
            input_ids = input_ids[: self.max_harmony_len]
            scale_vectors = scale_vectors[: self.max_harmony_len]
            chord_tones = chord_tones[: self.max_harmony_len]
            guide_vectors = guide_vectors[: self.max_harmony_len]
            tensions = tensions[: self.max_harmony_len]
            available_tensions = available_tensions[: self.max_harmony_len]
            avoids = avoids[: self.max_harmony_len]
            length = self.max_harmony_len

        pad_length = self.max_harmony_len - length

        input_ids += [[0] * 13] * pad_length
        scale_vectors += [[0] * 12] * pad_length
        chord_tones += [[0] * 12] * pad_length
        guide_vectors += [[0] * 12] * pad_length
        tensions += [[0] * 12] * pad_length
        available_tensions += [[0] * 12] * pad_length
        avoids += [[0] * 12] * pad_length

        attention_mask = [
            1 if i < length else 0
            for i in range(self.max_harmony_len)
        ]

        # ==================================
        # Tensor
        # ==================================

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "scale_vector": torch.tensor(
                scale_vectors, dtype=torch.float, device=self.device
            ).unsqueeze(0),
            "chord_tones_vector": torch.tensor(
                chord_tones, dtype=torch.float, device=self.device
            ).unsqueeze(0),
            "guide_vector": torch.tensor(
                guide_vectors, dtype=torch.float, device=self.device
            ).unsqueeze(0),
            "tension_vector": torch.tensor(
                tensions, dtype=torch.float, device=self.device
            ).unsqueeze(0),
            "available_tension_vector": torch.tensor(
                available_tensions, dtype=torch.float, device=self.device
            ).unsqueeze(0),
            "avoid_vector": torch.tensor(
                avoids, dtype=torch.float, device=self.device
            ).unsqueeze(0),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0),
        }

    def build_note_aligned(self, note_chords):
        """Build harmony memory from one already-resolved chord per note."""
        return self.build(note_chords, expand_bar_specs=False)
