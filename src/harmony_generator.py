import torch

from .jazz_features import JazzFeatures



class HarmonyGenerator:


    def __init__(self,max_harmony_len=128,device="cpu"):

        self.max_harmony_len=max_harmony_len
        self.device=device
        self.features = JazzFeatures()



    # ==========================================
    # Chord Normalize
    # Same as training Dataset
    # ==========================================

    def normalize_chord(self,chord):

        if chord is None or chord=="":
            return "N.C"


        chord=chord.strip()


        # Major symbol

        chord=chord.replace(
            "j",
            "^"
        )

        chord=chord.replace(
            "∆",
            "^"
        )


        # Minor

        if len(chord)>1:

            root_end=1

            if chord[1] in ["b","#"]:

                root_end=2


            root=chord[:root_end]

            rest=chord[root_end:]


            if rest.startswith("m"):

                rest=rest.replace(
                    "m",
                    "-",
                    1
                )


            chord=root+rest



        # Alter chord

        replacements={

            "alt":"b9#5",

            "ø":"-7b5"

        }


        for old,new in replacements.items():

            chord=chord.replace(
                old,
                new
            )


        return chord



    # ==========================================
    # Build Harmony Feature
    # ==========================================

    def build(self,chords):


        input_ids=[]


        scale_vectors=[]

        chord_tones=[]

        guide_vectors=[]

        tensions=[]

        available_tensions=[]

        avoids=[]



        for chord in chords:


            # same preprocessing as training

            chord=self.normalize_chord(
                chord
            )


            feature=self.features.analyze_chord(
                chord
            )



            input_ids.append(

                [

                    feature["chord_id"],

                    feature["root"],

                    feature["bass"],

                    feature.get(
                        "bass_interval",
                        0
                    ),

                    feature.get(
                        "inversion",
                        0
                    ),

                    feature["attribute"],


                    feature.get(
                        "function",
                        0
                    ),

                    feature.get(
                        "level",
                        0
                    ),

                    feature.get(
                        "scale",
                        0
                    ),

                    feature.get(
                        "duration",
                        0
                    ),

                    feature.get(
                        "beat",
                        0
                    ),

                    feature.get(
                        "section",
                        0
                    ),

                    feature.get(
                        "time",
                        0
                    )

                ]

            )



            scale_vectors.append(

                feature["scale_vector"]

            )


            chord_tones.append(

                feature["chord_tones"]

            )


            guide_vectors.append(

                feature["guide_tones"]

            )


            tensions.append(

                feature["tensions"]

            )


            available_tensions.append(

                feature["available_tensions"]

            )


            avoids.append(

                feature["avoid"]

            )



        # ==================================
        # Padding
        # ==================================

        length=len(input_ids)


        if length > self.max_harmony_len:


            input_ids=input_ids[:self.max_harmony_len]

            scale_vectors=scale_vectors[:self.max_harmony_len]

            chord_tones=chord_tones[:self.max_harmony_len]

            guide_vectors=guide_vectors[:self.max_harmony_len]

            tensions=tensions[:self.max_harmony_len]

            available_tensions=available_tensions[:self.max_harmony_len]

            avoids=avoids[:self.max_harmony_len]


            length=self.max_harmony_len



        pad_length=self.max_harmony_len-length



        input_ids += [

            [0]*13

        ]*pad_length



        scale_vectors += [

            [0]*12

        ]*pad_length



        chord_tones += [

            [0]*12

        ]*pad_length



        guide_vectors += [

            [0]*12

        ]*pad_length



        tensions += [

            [0]*12

        ]*pad_length



        available_tensions += [

            [0]*12

        ]*pad_length



        avoids += [

            [0]*12

        ]*pad_length




        attention_mask=[

            1 if i < length else 0

            for i in range(
                self.max_harmony_len
            )

        ]



        # ==================================
        # Tensor
        # ==================================


        return {


            "input_ids":

            torch.tensor(
                input_ids,
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0),



            "scale_vector":

            torch.tensor(
                scale_vectors,
                dtype=torch.float,
                device=self.device
            ).unsqueeze(0),



            "chord_tones_vector":

            torch.tensor(
                chord_tones,
                dtype=torch.float,
                device=self.device
            ).unsqueeze(0),



            "guide_vector":

            torch.tensor(
                guide_vectors,
                dtype=torch.float,
                device=self.device
            ).unsqueeze(0),



            "tension_vector":

            torch.tensor(
                tensions,
                dtype=torch.float,
                device=self.device
            ).unsqueeze(0),



            "available_tension_vector":

            torch.tensor(
                available_tensions,
                dtype=torch.float,
                device=self.device
            ).unsqueeze(0),



            "avoid_vector":

            torch.tensor(
                avoids,
                dtype=torch.float,
                device=self.device
            ).unsqueeze(0),



            "attention_mask":

            torch.tensor(
                attention_mask,
                dtype=torch.bool,
                device=self.device
            ).unsqueeze(0)

        }