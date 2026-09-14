import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.jazz_features import JazzFeatures



class JazzDataset(Dataset):


    def __init__(
        self,
        solo_dir,
        vocab_file,
        seq_length=512,
        stride=256,
        split="train",
        train_ratio=0.9,
        seed=42
    ):


        self.seq_length = seq_length


        # =====================
        # Vocabulary
        # =====================

        with open(
            vocab_file,
            "r",
            encoding="utf-8"
        ) as f:

            vocab=json.load(f)



        self.token_to_id=vocab["token_to_id"]


        self.pad_id=self.token_to_id["<PAD>"]

        self.bos_id=self.token_to_id["<BOS>"]

        self.eos_id=self.token_to_id["<EOS>"]

        self.unk_id=self.token_to_id["<UNK>"]



        self.feature=JazzFeatures()



        self.data=[]


        files=list(
            Path(solo_dir).glob("*.json")
        )


        print(
            "Total Songs:",
            len(files)
        )


        random.seed(seed)

        random.shuffle(files)



        split_idx=int(
            len(files)*train_ratio
        )



        if split=="train":

            files=files[:split_idx]

        else:

            files=files[split_idx:]



        print(
            split,
            "songs:",
            len(files)
        )



        # =====================
        # Load songs
        # =====================


        for file in files:


            try:

                with open(
                    file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    song=json.load(f)



                tokens=song["tokens"]



                # -----------------
                # Harmony
                # -----------------

                harmony=self.extract_harmony(
                    tokens
                )



                # -----------------
                # Melody tokens
                # -----------------

                ids=[]


                for token in tokens:


                    ids.append(
                        self.token_to_id.get(
                            token,
                            self.unk_id
                        )
                    )



                ids=[
                    self.bos_id
                ] + ids + [
                    self.eos_id
                ]



                # -----------------
                # sequence split
                # -----------------

                for start in range(
                    0,
                    len(ids)-seq_length,
                    stride
                ):


                    chunk=ids[
                        start:
                        start+seq_length
                    ]



                    self.data.append(
                        {
                            "ids":chunk,

                            "harmony":harmony
                        }
                    )



            except Exception as e:

                print(
                    "Error:",
                    file,
                    e
                )



        print(
            "Loaded sequences:",
            len(self.data)
        )



    # ==================================================
    # Extract Harmony
    # ==================================================

    def extract_harmony(self,tokens):
        harmony=[]

        function_map={
        "Tonic":0,
        "SubDominant":1,
        "Dominant":2
        }

        for token in tokens:
            if token.startswith("CHORD_"):
                chord=token.replace( "CHORD_", "")
                feature=self.feature.analyze_chord( chord)
                feature["function_id"] = function_map.get(feature["function"],0)
                harmony.append(feature)



        return harmony



    # ==================================================
    # Harmony Padding
    # ==================================================

    def process_harmony(self,harmony):
        MAX_HARMONY_LEN = 128
        harmony = harmony[: MAX_HARMONY_LEN]
        while len(harmony) < MAX_HARMONY_LEN:
            harmony.append({
            "chord_id":0,
            "root":0,
            "bass":0,
            "bass_interval":0,
            "inversion":0,
            "attribute":0,
            "function_id":0,
            "level":0,
            "scale":0,
            "duration":0,
            "beat":0,
            "section":0,
            "time":0,

            "scale_vector":[0]*12,

            "chord_tones":[0]*12,

            "guide_tones":[0]*12,

            "tensions":[0]*12,

            "available_tensions":[0]*12,

            "avoid":[0]*12
            })

        output={}
    # =====================
    # categorical harmony
    # =====================
        input_ids=[]


        for h in harmony:

            input_ids.append([
            h["chord_id"],
            h["root"],
            h["bass"],
            h["bass_interval"],
            h["inversion"],
            h["attribute"],
            h["function_id"],
            h["level"],
            h["scale"],
            h["duration"],
            h["beat"],
            h["section"],
            h["time"]
        ])


        output["input_ids"]=torch.tensor(
        input_ids,
        dtype=torch.long
    )


    # =====================
    # theory vectors
    # =====================

        output["scale_vector"]=torch.tensor(
        [h["scale_vector"] for h in harmony],
        dtype=torch.float32
    )


        output["chord_tones_vector"]=torch.tensor(
        [h["chord_tones"] for h in harmony],
        dtype=torch.float32
    )


        output["guide_vector"]=torch.tensor(
        [h["guide_tones"] for h in harmony],
        dtype=torch.float32
    )


        output["tension_vector"]=torch.tensor(
        [h["tensions"] for h in harmony],
        dtype=torch.float32
    )


        output["available_tension_vector"]=torch.tensor(
        [h["available_tensions"] for h in harmony],
        dtype=torch.float32
    )


        output["avoid_vector"]=torch.tensor(
        [h["avoid"] for h in harmony],
        dtype=torch.float32
    )


        return output



    # ==================================================
    # Dataset
    # ==================================================

    def __len__(self):

        return len(self.data)



    def __getitem__(
        self,
        index
    ):


        item=self.data[index]


        ids=item["ids"]



        if len(ids)<self.seq_length:

            ids += [
                self.pad_id
            ]*(self.seq_length-len(ids))



        else:

            ids=ids[:self.seq_length]



        input_ids=torch.tensor(
            ids[:-1],
            dtype=torch.long
        )


        labels=torch.tensor(
            ids[1:],
            dtype=torch.long
        )



        harmony=self.process_harmony(
            item["harmony"]
        )



        return {

            "input_ids":input_ids,

            "labels":labels,

            "harmony":harmony

        }