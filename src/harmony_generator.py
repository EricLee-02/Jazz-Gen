import torch

from .jazz_features import JazzFeatures


class HarmonyGenerator:

    def __init__(
        self,
        max_harmony_len=512,
        device="cpu"
    ):
        self.max_harmony_len = max_harmony_len
        self.device = device
        self.features = JazzFeatures()

        self.function_map = {
            "Tonic": 0,
            "SubDominant": 1,
            "Dominant": 2
        }

    # ==========================================
    # Chord Normalize
    # Same logic as training Dataset
    # ==========================================

    def normalize_chord(self, chord):

        if chord is None or chord == "":
            return "N.C"

        chord = str(chord).strip()

        if chord in [ "","N.C", "NC","None","nan"]:
            return "N.C"
        # Major symbol
        chord = chord.replace("j", "^")
        chord = chord.replace("∆", "^")
        # Minor symbol
        if len(chord) > 1:
            root_end = 1
            if chord[1] in ["b", "#"]:
                root_end = 2
            root = chord[:root_end]
            rest = chord[root_end:]
            if rest.startswith("m"):
                rest = "-" + rest[1:]
            chord = root + rest
        replacements = {
            "alt": "b9#5",
            "ø": "-7b5"
        }

        for old, new in replacements.items():
            chord = chord.replace(old, new)
        return chord

    # ==========================================
    # Convert one chord into encoder features
    # ==========================================

    def chord_to_feature(self, chord):
        chord = self.normalize_chord(chord)
        feature = self.features.analyze_chord(chord)
        function_id = self.function_map.get(feature.get("function"),0)
        categorical = [
            feature.get("chord_id", 0),
            feature.get("root", 0),
            feature.get("bass", 0),
            feature.get("bass_interval", 0),
            feature.get("inversion", 0),
            feature.get("attribute", 0),
            function_id,
            feature.get("level", 0),
            feature.get("scale", 0),
            feature.get("duration", 0),
            feature.get("beat", 0),
            feature.get("section", 0),
            feature.get("time", 0)
        ]

        return {
            "input_ids": categorical,
            "scale_vector":feature.get("scale_vector",[0] * 12),
            "chord_tones_vector":feature.get("chord_tones",[0] * 12),
            "guide_vector":feature.get("guide_tones", [0] * 12),
            "tension_vector":feature.get( "tensions",[0] * 12),
            "available_tension_vector":feature.get( "available_tensions",[0] * 12),
            "avoid_vector": feature.get("avoid",[0] * 12),
            }

    # ==========================================
    # Core builder
    #
    # IMPORTANT:
    # chord_events means:
    #
    # [
    #   chord_for_note_1,
    #   chord_for_note_2,
    #   chord_for_note_3,
    #   ...
    # ]
    #
    # If called from build(), it can also simply
    # be a chord progression.
    # ==========================================

    def _build_from_events(self, chord_events):
        input_ids = []
        scale_vectors = []
        chord_tones = []
        guide_vectors = []
        tensions = []
        available_tensions = []
        avoids = []
        for chord in chord_events:
            feature = self.chord_to_feature(chord)
            input_ids.append(feature["input_ids"])
            scale_vectors.append(feature["scale_vector"])
            chord_tones.append( feature["chord_tones_vector"])
            guide_vectors.append(feature["guide_vector"])
            tensions.append(feature["tension_vector"])
            available_tensions.append( feature["available_tension_vector"])
            avoids.append(feature["avoid_vector"])

        # ==================================
        # Truncate
        # ==================================
        length = len(input_ids)
        if length > self.max_harmony_len:
            # For generation we normally care more
            # about recent note-level harmony context.
            input_ids = input_ids[ -self.max_harmony_len:]
            scale_vectors = scale_vectors[-self.max_harmony_len:]
            chord_tones = chord_tones[-self.max_harmony_len:]
            guide_vectors = guide_vectors[-self.max_harmony_len:]
            tensions = tensions[-self.max_harmony_len:]
            available_tensions = available_tensions[-self.max_harmony_len:]
            avoids = avoids[ -self.max_harmony_len:]
            length = self.max_harmony_len
        # ==================================
        # Padding
        # ==================================
        pad_length = (self.max_harmony_len - length)
        input_ids += [[0] * 13 for _ in range(pad_length) ]
        scale_vectors += [[0] * 12 for _ in range(pad_length)]
        chord_tones += [[0] * 12 for _ in range(pad_length) ]
        guide_vectors += [ [0] * 12 for _ in range(pad_length)]
        tensions += [ [0] * 12 for _ in range(pad_length)]
        available_tensions += [[0] * 12 for _ in range(pad_length)]
        avoids += [ [0] * 12 for _ in range(pad_length) ]
        attention_mask = [ 1 if i < length else 0 for i in range( self.max_harmony_len)]
        # ==================================
        # Tensor
        # ==================================
        return {
            "input_ids": torch.tensor(input_ids,  dtype=torch.long, device=self.device).unsqueeze(0),
            "scale_vector": torch.tensor( scale_vectors,dtype=torch.float32, device=self.device).unsqueeze(0),
            "chord_tones_vector":torch.tensor(chord_tones, dtype=torch.float32, device=self.device ).unsqueeze(0),
            "guide_vector":torch.tensor( guide_vectors, dtype=torch.float32,device=self.device).unsqueeze(0),
            "tension_vector":torch.tensor( tensions, dtype=torch.float32,device=self.device).unsqueeze(0),
            "available_tension_vector":torch.tensor(available_tensions,dtype=torch.float32, device=self.device).unsqueeze(0),
            "avoid_vector":torch.tensor( avoids,dtype=torch.float32, device=self.device).unsqueeze(0),
            "attention_mask": torch.tensor(  attention_mask, dtype=torch.bool, device=self.device ).unsqueeze(0)
            }

    # ==========================================
    # Old mode
    #
    # One chord -> one harmony event
    # Mainly for compatibility / testing
    # ==========================================

    def build(self, chords):
        return self._build_from_events(chords)

    # ==========================================
    # NOTE-TO-CHORD mode
    #
    # One generated note -> one harmony event
    #
    # This is the mode that matches the new
    # training Dataset.
    # ==========================================

    def build_note_aligned(self, note_chords):
        if not note_chords:
            note_chords = ["N.C"]
        return self._build_from_events(note_chords)

    # ==========================================
    # Resolve chord from current BAR
    #
    # Current assumption:
    # one item in chords = one bar.
    #
    # bar is zero-based:
    # BAR_0 -> chords[0]
    # BAR_1 -> chords[1]
    # ...
    # ==========================================

    def chord_for_bar(self, chords, bar):
        if not chords:
            return "N.C"
        try:
            bar = int(bar)
        except (TypeError, ValueError):
            bar = 0
        if bar < 0:
            bar = 0
        if bar >= len(chords):
            return chords[-1]
        return chords[bar]