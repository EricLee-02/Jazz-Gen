import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset

from .jazz_features import JazzFeatures


class JazzGenerationDataset(Dataset):

    def __init__(self,
                 solo_dir,
        vocab_file,
        seq_length=512,
        max_harmony_len=512,
        stride=256,
        split = "train",
        train_ratio = 0.9,
        seed =42
    ):

        self.seq_length = seq_length
        self.max_harmony_len = max_harmony_len
        self.stride = stride

        self.features = JazzFeatures()

        # ==================================================
        # New Melody Note Grammar
        # POSITION / SUBTATUM are intentionally excluded
        # ==================================================
        self.required_note_prefixes = (
            "SECTION_",
            "BAR_",
            "PERIOD_",
            "BEAT_",
            "DIVISION_",
            "TATUM_",
            "PITCH_",
            "DURATION_",
            "VELOCITY_",
        )

        self.optional_note_prefixes = (
            "ARTIC_",
            "MICRO_",
        )
        self.note_prefixes = (self.required_note_prefixes+ self.optional_note_prefixes)

        # ==========================
        # Load Melody Vocabulary
        # ==========================

        with open(vocab_file,"r",encoding="utf8") as f:
            vocab = json.load(f)
        self.token_to_id = vocab["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab["id_to_token"].items()}
        self.tempo_vocab = {}
        for token in self.token_to_id:
            if token.startswith('AVGTEMPO_'):
                try:
                    value = float(token.replace('AVGTEMPO_',"",1))
                    self.tempo_vocab[token] = value
                except ValueError:
                    continue
        if not self.tempo_vocab:
            raise ValueError("No AVGTEMPO_tokens found in vocabulary")

        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

        # ==========================
        # Load Solo Files
        # ==========================
        all_files = sorted(Path(solo_dir).glob("*.json"))

    # ==========================
    # Remove songs without chord
    # BEFORE train/val split
    # ==========================
        valid_files = []
        skipped_no_chord = 0
        for file in all_files:
            with open(file, "r", encoding="utf8") as f:
                data = json.load(f)
            tokens = data["tokens"]
            has_chord = any(token.startswith("CHORD_") for token in tokens)
            if has_chord:
                valid_files.append(file)
            else:
                skipped_no_chord += 1

        print("Valid harmony songs:", len(valid_files))
        print("Skipped no-chord songs:", skipped_no_chord)

    # ==========================
    # Song-level train/val split
    # ==========================
        rng = random.Random(seed)
        rng.shuffle(valid_files)
        split_idx = int( len(valid_files) * train_ratio)

        if split == "train":
            self.files = valid_files[:split_idx]
        elif split == "val":
            self.files = valid_files[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        print( f"{split} songs:",len(self.files))
        self.samples = []
        skipped_prompt = 0
        skipped_no_notes = 0
        skipped_no_chords = 0
        skipped_no_chords_songs = 0

        for file in self.files:
            with open(file, "r", encoding="utf8") as f:
                data = json.load(f)
            tokens = data["tokens"]
            has_chord = any(token.startswith("CHORD_") for token in tokens)
            if not has_chord:
                skipped_no_chords_songs +=1
                continue

            # --------------------------
            # Global conditions
            # --------------------------
            key_token, tempo_token = ( self.extract_global_prompt(tokens))
            # We want every training chunk to have
            # exactly the same global condition format
            # as generation:
            #
            # <BOS>
            # KEY_xxx
            # AVGTEMPO_xxx
            #
            if (key_token is None or tempo_token is None):
                skipped_prompt += 1
                continue
            # --------------------------
            # Sliding windows
            # -------------------------
            for start in range(0,len(tokens),self.stride):
                segment = tokens[start:start + self.seq_length]
                harmony_events = self.extract_note_aligned_chords(segment)
                if len(segment) <= 50:
                    continue

                # ==================================
                # Keep complete melody notes only
                # ==================================
                melody_tokens = (self.extract_complete_note_tokens(segment))
                if not melody_tokens:
                    skipped_no_notes += 1
                    continue

                # ==================================
                # Harmony aligned to this chunk
                # ==================================

                chords = self.extract_note_aligned_chords(segment=segment )
                if not chords:
                    skipped_no_chords += 1
                    continue

                # Only a chunk containing the real
                # end of the song should learn EOS
                is_last = ( "<EOS>" in segment or start + self.seq_length>= len(tokens))
                self.samples.append({
                    "melody_tokens":melody_tokens,
                    "harmony_events": harmony_events,
                    "chords":chords,
                    "key_token": key_token,
                    "tempo_token":tempo_token,
                    "is_last":is_last,

                })

        print("Solo files:",len(self.files))
        print("Skipped songs with no chord:", skipped_no_chords_songs)
        print("Generation samples:",len(self.samples))
        print("Skipped missing key/tempo:",skipped_prompt)
        print("Skipped no complete notes:",skipped_no_notes)
        print("Skipped no harmony:",skipped_no_chords)


    def _resolve_key_token(self, key_token):
    # 1. Exact match
        if key_token in self.token_to_id:
            return key_token
    # 2. Invalid format
        if (not key_token.startswith("KEY_")):
            return "<UNK>"
        body = key_token[4:]
    # e.g. KEY_Ab
        if "-" not in body:
            return "<UNK>"
        root, mode = body.split("-", 1)
        if mode == "chorm":
            mode = "maj"
        candidate = f"KEY_{root}-{mode}"
        if candidate in self.token_to_id:
            return candidate
    # 4. Final fallback
        return "<UNK>"

    def match_tempo_token(self,tempo_token):
        if tempo_token in self.token_to_id:
            return tempo_token
        try :
            tempo_value = float(tempo_token.replace('AVGTEMPO_',"",1))
        except ValueError:
            raise ValueError( f"Invalid tempo token: {tempo_token}")
        candidates = []
        for token in self.token_to_id:
            if not token.startswith('AVGTEMPO_'):
                continue
            try:
                value = float(token.replace('AVGTEMPO_','',1))
                candidates.append((abs(value-tempo_value),token))
            except ValueError:
                continue
        if not candidates:
            raise ValueError("No AVGTEMPO tokens found in vocabulary.")
        
        _,nearest_token = min(candidates,key=lambda x:x[0])

        return nearest_token

    # ==================================================
    # Dataset Length
    # ==================================================

    def __len__(self):
        return len(self.samples)

    # ==================================================
    # Extract Global Prompt
    # ==================================================
    def extract_global_prompt( self,tokens):
        key_token = None
        tempo_token = None
        for token in tokens:
            if ( token.startswith("KEY_") and key_token is None):
                key_token = token
            elif (token.startswith("AVGTEMPO_") and tempo_token is None):
                tempo_token = token
            if (key_token is not None and tempo_token is not None ):
                break
        if tempo_token is not None:
            tempo_token = self.match_tempo_token(tempo_token)

        return (key_token, tempo_token )

    # ==================================================
    # Extract Complete Melody Notes
    #
    # New note format:
    #
    # BAR
    # PERIOD
    # BEAT
    # DIVISION
    # TATUM
    # PITCH
    # DURATION
    # VELOCITY
    # [ARTIC]
    # [MICRO]
    #
    # POSITION and SUBTATUM are ignored.
    # Harmony/header tokens are also ignored.
    # ==================================================

    def extract_complete_note_tokens( self,tokens):
        complete_tokens = []
        current_note = None
        for token in tokens:
            # Every note begins with BAR
            if token.startswith("SECTION_"):
                if current_note is not None:
                    if self.is_complete_note(current_note):
                        complete_tokens.extend(current_note)
                current_note = [token]
                continue 

            # Ignore everything before first BAR
            if current_note is None:
                continue
            # Only retain tokens belonging to
            # the new melody grammar
            if token.startswith(self.note_prefixes):
                current_note.append( token)
        # Final note
        if ( current_note is not None and self.is_complete_note( current_note )):
            complete_tokens.extend(current_note )

        return complete_tokens

    # ==================================================
    # Check Complete Note
    # ==================================================

    def is_complete_note( self, note_tokens ):
        for prefix in (self.required_note_prefixes ):
            if not any( token.startswith(prefix) for token in note_tokens):
                return False
        return True

    # ==================================================
    # Chord Normalize
    # ==================================================

    def normalize_chord(  self,chord):

        """
        Convert WJazzD chord notation
        to JazzFeatures notation.

        Examples:

        Cj7  -> C^7
        Cm7  -> C-7
        Calt -> C7b9#5
        """
        if ( chord is None or chord == ""):
            return None
        chord = chord.strip()
        # ----------------------
        # Major
        # ----------------------
        chord = chord.replace( "j", "^")
        chord = chord.replace("∆", "^")
        # ----------------------
        # Minor
        # ----------------------
        if len(chord) > 1:
            root_end = 1
            if chord[1] in [ "b", "#"]:
                root_end = 2
            root = chord[:root_end]
            rest = chord[root_end:]
            if rest.startswith("m"):
                rest = rest.replace("m","-",1)
            chord = root + rest
        # ----------------------
        # Alter chords
        # ----------------------

        replacements = { "alt": "b9#5", "ø": "-7b5",}
        for old, new in ( replacements.items()):
            chord = chord.replace(old, new)
        return chord

    # ==================================================
    # Extract Harmony for Current Segment
    # ==================================================

    def extract_note_aligned_chords(self,segment):
        aligned = []
        current_chord = None
        inside_note =False
        # ==================================
        # Chord changes inside segment
        # ==================================

        for token in segment:
            if token.startswith("CHORD_" ):
                chord = token.replace("CHORD_","", 1)
                current_chord = (self.normalize_chord(chord))
            if token.startswith("BAR_"):
                inside_note = True
            if token.startswith("PITCH_") and inside_note:
                if current_chord:
                    aligned.append(current_chord)
                    inside_note = False
        return aligned
    def split_note_events(self, note_tokens):
        events = []
        current = []
        for token in note_tokens:
            if token.startswith("SECTION_"):
                current.append(token)
            elif token.startswith("BAR_"):
                if current and self.is_complete_note(current):
                    events.append(current)
                current = [token]
            else:
                if current:
                    current.append(token)
    # final note
        if current:
            if self.is_complete_note(current):
                events.append(current)
        return events
        

    # ==================================================
    # Chord -> Harmony Features
    # ==================================================

    def build_harmony(self,harmony_events):

        input_ids = []
        scale_vectors = []
        chord_tones = []
        guide_tones = []
        tensions = []
        available_tensions = []
        avoids = []
        function_map = {
            "Tonic":0,
            "SubDominant":1,
            "Dominant":2
            }

        for chord in harmony_events:
            if chord is None:
                continue
            chord = self.features.normalize_chord_map(chord)
            try:
                feature = (self.features.analyze_chord(chord))
            except Exception:
                continue
            function_id = (function_map.get(feature.get("function"),0))

            # ==================================
            # Exactly 13 categorical features
            # ==================================

            input_ids.append([
                feature["chord_id"],
                feature["root"],
                feature["bass"],
                feature.get("bass_interval",0),
                feature.get("inversion",0),
                feature["attribute"],
                function_id,
                feature.get("level",0),
                feature.get("scale",0),
                feature.get("duration",0),
                feature.get("beat",0),
                feature.get("section",0),
                feature.get( "time",0),
                ])

            scale_vectors.append(feature["scale_vector"])
            chord_tones.append(feature["chord_tones"])
            guide_tones.append(feature["guide_tones"])
            tensions.append(feature["tensions"])
            available_tensions.append(feature["available_tensions" ])
            avoids.append(feature["avoid"] )

        # ==================================
        # Ensure valid harmony
        # ==================================

        if not input_ids:
            raise RuntimeError(
                f"No valid harmony features "
                f"could be built from: {harmony_events}"
            )

        # ==================================
        # Truncate
        # ==================================

        if (len(input_ids)> self.max_harmony_len):
            input_ids = input_ids[:self.max_harmony_len]
            scale_vectors = (scale_vectors[:self.max_harmony_len])
            chord_tones = (chord_tones[:self.max_harmony_len])
            guide_tones = ( guide_tones[:self.max_harmony_len])
            tensions = (tensions[ :self.max_harmony_len])
            available_tensions = (available_tensions[:self.max_harmony_len ])
            avoids = (avoids[:self.max_harmony_len])
        length = len(input_ids)
        pad_length = ( self.max_harmony_len- length)

        attention_mask = ([1] * length+ [0] * pad_length)

        # ==================================
        # Padding
        # ==================================
        input_ids += [[0] * 13 for _ in range(pad_length)]
        scale_vectors += [[0] * 12 for _ in range( pad_length )]
        chord_tones += [[0] * 12 for _ in range(pad_length)]
        guide_tones += [[0] * 12 for _ in range(pad_length)]
        tensions += [[0] * 12 for _ in range( pad_length)]

        available_tensions += [ [0] * 12 for _ in range(pad_length)]
        avoids += [[0] * 12 for _ in range( pad_length)]

        return {
            "input_ids":torch.tensor( input_ids,dtype=torch.long),
            "scale_vector": torch.tensor( scale_vectors, dtype=torch.float ),
            "chord_tones_vector": torch.tensor( chord_tones,dtype=torch.float ),
            "guide_vector": torch.tensor( guide_tones,dtype=torch.float),
            "tension_vector":torch.tensor(  tensions, dtype=torch.float ),
            "available_tension_vector":torch.tensor( available_tensions, dtype=torch.float),
            "avoid_vector": torch.tensor(  avoids, dtype=torch.float ),
            "attention_mask": torch.tensor( attention_mask, dtype=torch.bool ),
        }

    # ==================================================
    # Melody Processing
    # ==================================================

    def build_melody(self,note_tokens,key_token,tempo_token, is_last):
        key_token = self._resolve_key_token(key_token)
        if tempo_token not in self.token_to_id:
            raise ValueError(f"Tempo token not in vocabulary: "f"{tempo_token}")

        # ==================================
        # Same prompt used at inference
        # ==================================

        prompt_tokens = [ "<BOS>",key_token,tempo_token,]
        max_total_tokens = self.seq_length +1
        sequence_tokens = list(prompt_tokens)
        note_events = self.split_note_events(note_tokens)
        included_notes = 0
   # ==================================
    # Add notes ONLY if the entire note fits
    # ==================================

        for event in note_events:

        # Never cut a note in the middle
            if ( len(sequence_tokens)+ len(event) > max_total_tokens):
                break
            sequence_tokens.extend(event)
            included_notes += 1

    # ==================================
    # EOS
    # ==================================
        all_notes_included = (included_notes== len(note_events))

    # Only train EOS when:
    #
    # 1. this is actually the end of the song
    # 2. all notes from this segment were retained
    # 3. EOS itself fits
    #
        if (is_last and all_notes_included and len(sequence_tokens) + 1 <= max_total_tokens):
            sequence_tokens.append( "<EOS>" )

    # ==================================
    # Token -> ID
    # ==================================

        ids = [self.token_to_id.get(token, self.unk_id) for token in sequence_tokens]
        input_ids = ids[:-1]
        target_ids = ids[1:]

    # ==================================
    # Ignore prompt prediction loss
    #
    # <BOS> -> KEY       ignore
    # KEY   -> TEMPO     ignore
    # TEMPO -> BAR       train
    # ==================================

        prompt_loss_positions = ( len(prompt_tokens) - 1)
        for i in range( min(prompt_loss_positions,len(target_ids))):
            target_ids[i] = -100

    # ==================================
    # Padding
    # ==================================

        pad_len = (self.seq_length - len(input_ids))

        if pad_len > 0:
            input_ids += ([self.pad_id]* pad_len)
            target_ids += ([-100]* pad_len)

    # Defensive check
        if len(input_ids) != self.seq_length:
            raise RuntimeError(
            f"input length error: "
            f"{len(input_ids)} != "
            f"{self.seq_length}"
        )

        if len(target_ids) != self.seq_length:
            raise RuntimeError(
            f"target length error: "
            f"{len(target_ids)} != "
            f"{self.seq_length}"
        )

        attention_mask = [
        token_id != self.pad_id
        for token_id in input_ids
    ]

        return {
        "input_ids": torch.tensor( input_ids,dtype=torch.long),
        "target_ids":torch.tensor(target_ids, dtype=torch.long ),
        "attention_mask": torch.tensor( attention_mask, dtype=torch.bool )
          }


    # ==================================================
    # Get Item
    # ==================================================

    def __getitem__(self,index):
        sample = self.samples[index]
        # ==================================
        # Harmony
        # ==================================
        harmony = (self.build_harmony(sample["harmony_events"]))
        # ==================================
        # Melody
        # ==================================

        melody = (
            self.build_melody(
                note_tokens= sample["melody_tokens"],
                key_token=sample["key_token" ],
                tempo_token=sample["tempo_token"],
                is_last=sample[ "is_last" ],
                ) )

        return {
            "harmony": harmony,
            "melody_input":  melody["input_ids"],
            "melody_target": melody["target_ids"],
            "attention_mask": melody["attention_mask" ],
            }