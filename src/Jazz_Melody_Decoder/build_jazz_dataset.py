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

    def extract_harmony(
        self,
        tokens
    ):


        harmony=[]


        for token in tokens:


            if token.startswith(
                "CHORD_"
            ):


                chord=token.replace(
                    "CHORD_",
                    ""
                )


                feature=self.feature.analyze_chord(
                    chord
                )


                harmony.append(
                    feature
                )



        return harmony



    # ==================================================
    # Harmony Padding
    # ==================================================

    def process_harmony(
        self,
        harmony,
        max_len=128
    ):


        keys=[

            "chord_id",

            "root",

            "bass",

            "bass_interval",

            "inversion",

            "attribute",

            "function",

            "level",

            "scale",

            "duration",

            "beat",

            "section",

            "time",

            "scale_vector",

            "chord_tones",

            "guide_tones",

            "tensions",

            "available_tensions",

            "avoid"

        ]


        output={}



        for key in keys:


            values=[
                h[key]
                for h in harmony
            ]



            if len(values)<max_len:


                pad_len=max_len-len(values)



                if len(values)>0 and isinstance(
                    values[0],
                    list
                ):


                    values += [
                        [0]*len(values[0])
                    ]*pad_len


                else:

                    values += [
                        0
                    ]*pad_len



            else:

                values=values[:max_len]



            dtype=torch.float if key in [

                "scale_vector",
                "chord_tones",
                "guide_tones",
                "tensions",
                "available_tensions",
                "avoid"

            ] else torch.long



            output[key]=torch.tensor(
                values,
                dtype=dtype
            )



        output["attention_mask"]=torch.tensor(

            [
                1
                if i<len(harmony)
                else 0

                for i in range(max_len)
            ],

            dtype=torch.bool

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