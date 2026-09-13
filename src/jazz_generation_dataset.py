import json
import torch

from pathlib import Path
from torch.utils.data import Dataset

from jazz_features import JazzFeatures



class JazzGenerationDataset(Dataset):


    def __init__(self,solo_dir,vocab_file,seq_length=512,max_harmony_len=128,stride=256):
        self.seq_length = seq_length
        self.max_harmony_len = max_harmony_len
        self.stride = stride


        # ==========================
        # Load Melody Vocabulary
        # ==========================

        with open(vocab_file,"r",encoding="utf8") as f:
            vocab=json.load(f)
        self.token_to_id=vocab["token_to_id"]
        self.id_to_token={int(k):v for k,v in vocab["id_to_token"].items()}

        self.pad_id=self.token_to_id["<PAD>"]
        self.bos_id=self.token_to_id["<BOS>"]
        self.eos_id=self.token_to_id["<EOS>"]
        self.unk_id=self.token_to_id["<UNK>"]

        # ==========================
        # Load Solo Files
        # ==========================

        self.files=list(Path(solo_dir).glob("*.json"))
        self.samples = []
        for file in self.files:
            with open(file,"r") as f:
                data = json.load(f)
            tokens = data["tokens"]
            for i in range(0,len(tokens),self.stride):
                segment = tokens[i:i+self.seq_length]
                if len(segment)>50:
                    self.samples.append({
                        "tokens":segment
                    })
        print("Solo files:",len(self.files))



    # ==================================
    # Dataset Length
    # ==================================

    def __len__(self):
        return len(self.samples)


    # ==================================
    # Chord Normalize
    # ==================================

    def normalize_chord(self,chord):

        """
        Convert WJazzD chord notation
        to JazzFeatures notation
        Examples:
        Cj7->C^7
        Cm7->C-7
        Calt-> C7b9#5

        """

        if chord is None:
            return None
        chord=chord.strip()

        # ----------------------
        # Major
        # ----------------------
        chord=chord.replace("j","^")
        chord=chord.replace("∆","^")
        # ----------------------
        # Minor
        # ----------------------

        if len(chord)>1:
            root_end=1
            if chord[1] in ["b","#"]:
                root_end=2
            root=chord[:root_end]
            rest=chord[root_end:]

            if rest.startswith("m"):
                rest=rest.replace("m","-",1)
                chord=root+rest
        # ----------------------
        # Alter chords
        # ----------------------

        replacements={
            "alt":"b9#5",
            "ø":"-7b5",
        }

        for old,new in replacements.items():
            chord=chord.replace(old,new)
        return chord


    # ==================================
    # Extract chord token
    # ==================================

    def extract_chords(self,tokens):
        chords=[]
        for token in tokens:
            if token.startswith("CHORD_"):
                chord=token.replace("CHORD_","")
                chord=self.normalize_chord(chord)
                chords.append(chord)
        return chords

    # ==================================
    # Chord -> Harmony Feature
    # ==================================

    def build_harmony(self,chords):
        input_ids=[]
        scale_vectors=[]
        chord_tones=[]
        guide_tones=[]
        tensions=[]
        available_tensions=[]
        avoids=[]
        for chord in chords:
            try:
                feature=JazzFeatures.analyze_chord(chord)
            except Exception:
                continue

            input_ids.append([feature["chord_id"],feature["root"],feature["bass"]])
            scale_vectors.append(feature["scale_vector"])
            chord_tones.append(feature["chord_tones"])
            guide_tones.append(feature["guide_tones"])
            tensions.append(feature["tensions"])
            available_tensions.append(feature["available_tensions"])
            avoids.append(feature["avoid"])



        # padding harmony sequence
        length=len(input_ids)

        if length > self.max_harmony_len:
            input_ids=input_ids[:self.max_harmony_len]
            scale_vectors=scale_vectors[:self.max_harmony_len]
            chord_tones=chord_tones[:self.max_harmony_len]
            guide_tones=guide_tones[:self.max_harmony_len]
            tensions=tensions[:self.max_harmony_len]
            available_tensions=available_tensions[:self.max_harmony_len]
            avoids=avoids[:self.max_harmony_len]
        length = len(input_ids)
        pad_length=self.max_harmony_len-length
        attention_mask = [1 if i< length else 0 for i in range(self.max_harmony_len)]
        input_ids += [0]*pad_length
        scale_vectors += [[0]*12]*pad_length
        chord_tones += [[0]*12]*pad_length
        guide_tones += [[0]*12]*pad_length
        tensions += [[0]*12]*pad_length
        available_tensions += [[0]*12]*pad_length
        avoids += [[0]*12]*pad_length
        return {
            "input_ids":torch.tensor(input_ids,dtype=torch.long),
            "scale_vector":torch.tensor(scale_vectors,dtype=torch.float),
            "chord_tones_vector":torch.tensor(chord_tones,dtype=torch.float),
            "guide_vector":torch.tensor(guide_tones,dtype=torch.float),
            "tension_vector":torch.tensor(tensions,dtype=torch.float),
            "available_tension_vector":torch.tensor(available_tensions,dtype=torch.float),
            "avoid_vector":torch.tensor(avoids,dtype=torch.float),
            "attention_mask": torch.tensor(attention_mask,dtype=torch.bool)
        }



    # ==================================
    # Melody Processing
    # ==================================

    def build_melody(self,tokens):
        ids=[]
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))

        ids=[self.bos_id]+ids+[self.eos_id]
        ids=ids[:self.seq_length]
        input_ids=ids[:-1]
        target_ids=ids[1:]
        pad_len=self.seq_length-len(input_ids)
        input_ids += [self.pad_id]*pad_len
        target_ids += [-100]*pad_len
        attention_mask=[1 if x!=self.pad_id else 0 for x in input_ids]
        return {
            "input_ids":torch.tensor(input_ids, dtype=torch.long),
            "target_ids":torch.tensor(target_ids,dtype=torch.long),
            "attention_mask":torch.tensor(attention_mask,dtype=torch.bool)
        }

    # ==================================
    # Get Item
    # ==================================

    def __getitem__(self, index):
        sample=self.samples[index]

        # solo token
        tokens=sample["tokens"]
        # harmony
        chords=self.extract_chords(tokens)
        harmony=self.build_harmony(chords)

        # melody
        melody=self.build_melody(tokens)

        return {
            "harmony":harmony,
            "melody_input":melody["input_ids"],
            "melody_target":melody["target_ids"],
            "attention_mask":melody["attention_mask"]
        }