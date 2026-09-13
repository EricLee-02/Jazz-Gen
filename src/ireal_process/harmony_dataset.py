import json
import torch
import random
import os
from config import HARMONY_TOKEN_DIR,HARMONY_VOCAB_DIR
from torch.utils.data import Dataset
from jazz_features import JazzFeatures

class HarmonyDataset(Dataset):

    def __init__(self,json_file,vocab_dir,seq_length=128,split="train",train_ratio=0.9,seed = 42):
        self.seq_length = seq_length

        # ==========================
        # Load vocabulary
        # ==========================

        self.chord_vocab = self.load_vocab(vocab_dir + "/chord_vocab.json")
        self.duration_vocab = self.load_vocab(vocab_dir + "/duration_vocab.json")
        self.beat_vocab = self.load_vocab(vocab_dir + "/beat_vocab.json")
        self.section_vocab = self.load_vocab(vocab_dir + "/section_vocab.json")
        self.time_vocab = self.load_vocab(vocab_dir + "/time_signature_vocab.json")

        self.pad_id = self.chord_vocab["<PAD>"]
        self.unk_id = self.chord_vocab["<UNK>"]
        self.mask_id = self.chord_vocab["<MASK>"]
        # ==========================
        # Load songs
        # ==========================


        split_idx = int(len(data)*train_ratio)

        if split == "train":
            data = data[:split_idx]

        elif split == "val":
            data=data[split:]

        else:
            raise ValueError("split must be train or val")

        with open(json_file,"r",encoding="utf8") as f:
            songs=json.load(f)
        random.seed(seed)
        random.shuffle(songs)
        self.songs = data
        print("Total songs:",len(songs))



        # ==========================
        # Song -> sequence
        # 一首歌一个sequence
        # ==========================

        self.token_data = []
        self.scale_data = []
        self.chord_tone_data = []
        self.guide_tone_data = []
        self.tension_data = []
        self.available_tension_data = []
        self.avoid_data = []


        valid_song=0
        total_events=0


        for song in songs:
            tokens,scales,chord_tones,guide,tensions,available_tension,avoid=self.song_to_sequence(song)

            if len(tokens)==0:
                continue

            self.token_data.append(tokens)
            self.scale_data.append(scales)
            self.chord_tone_data.append(chord_tones)
            self.guide_tone_data.append(guide)
            self.tension_data.append(tensions)
            self.available_tension_data.append(available_tension)
            self.avoid_data.append(avoid)
            valid_song += 1
            total_events += len(tokens)

        print("Valid songs:",valid_song)
        print("Total events:",total_events)
        print("Dataset sequences:",len(self.token_data))



    # =================================================
    # Load vocab
    # =================================================

    def load_vocab(self,path):

        with open(path,"r",encoding="utf8") as f:
            data=json.load(f)
        return data["token_to_id"]

    # =================================================
    # Section lookup
    # =================================================

    def get_section(self,bar,sections):
        for sec in sections:
            if (sec["start_bar"]<=bar<=sec["end_bar"]):
                return sec["name"]

        return "<UNK>"
    
    def mask_sequence(self,tokens,mask_prob = 0.15):
        input_tokens = []
        labels = []
        for token in tokens:
            chord_id = token[0]
            if random.random() < mask_prob:
                new_token = token.copy()
                new_token[0] = self.mask_id
                input_tokens.append(new_token)
                labels.append(chord_id)
            else:
                input_tokens.append(token)
                labels.append(-100)
        return input_tokens,labels


    # =================================================
    # Song -> sequence
    # =================================================


    def song_to_sequence(self,song):
        token_sequence = []
        scale_sequence = []
        chord_tone_sequence = []
        guide_vector_sequence = []
        tensions_vector_sequence = []
        available_tensions_sequence = []
        avoid_sequence = []
        sections=(song.get("form",{}).get("sections",[]))
        time=song.get("time_signature",[4,4])
        time_token=f"{time[0]}/{time[1]}"

        time_id=self.time_vocab.get(time_token,self.time_vocab["<UNK>"])
        events=song.get("events",[])
        for event in events:
            chord=event.get("raw_symbol")
            if chord is None:
                continue
            embedding = event.get("embedding",{})

            #chord token
            chord_id=self.chord_vocab.get(chord,self.unk_id)
            root_id = embedding.get("root_pitch_class",0)
            bass_id = embedding.get("bass_pitch_class",0)
            bass_interval_id = embedding.get("bass_interval",0)
            inversion_id = embedding.get("inversion",0)
            attribute_id = embedding.get("attribute_id",0)
            function_id = embedding.get("function_id",0)
            roman_numeral = embedding.get("roman_numeral","I")
            roman_id = JazzFeatures.ROMAN_REVERSE_MAP.get(roman_numeral,0)
            # cadence_id = embedding.get("cadence_id",0)
            scale_id = embedding.get("scale_id",20)
            scale_vector = embedding.get("scale_vector",[0]*12)
            # style_id = embedding.get("style_id",10)
            # rhythm_id = embedding.get("rhythm_id",8)
            # phrase_id = embedding.get("phrase_id",0)
            duration=str(event.get( "duration","<UNK>"))
            duration_id=self.duration_vocab.get(duration,self.duration_vocab["<UNK>"])
            beat = str(event.get("beat_start","<UNK>"))
            beat_id=self.beat_vocab.get(beat,self.beat_vocab["<UNK>"])
            bar=event.get( "bar",0)
            section=self.get_section(bar,sections)
            section_id=self.section_vocab.get(section,self.section_vocab["<UNK>"])
            token=[
                chord_id,
                root_id,
                bass_id,
                bass_interval_id,
                inversion_id,
                attribute_id,
                function_id,
                roman_id,
                # cadence_id,
                scale_id,
                # style_id,
                # rhythm_id,
                # phrase_id,
                duration_id,
                beat_id,
                section_id,
                time_id
            ]
            chord_tone_vector = self.interval_to_vectors(embedding.get("chord_tones",[]))
            guide_vector = self.interval_to_vectors(embedding.get("guide_tones",[]))
            tension_vector = self.interval_to_vectors(embedding.get("tensions",[]))
            available_tension_vector = self.interval_to_vectors(embedding.get("available_tensions",[]))
            avoid_vector = self.interval_to_vectors(embedding.get("avoid",[]))

            token_sequence.append(token)
            scale_sequence.append(scale_vector)
            chord_tone_sequence.append(chord_tone_vector)
            guide_vector_sequence.append(guide_vector)
            tensions_vector_sequence.append(tension_vector)
            available_tensions_sequence.append(available_tension_vector)
            avoid_sequence.append(avoid_vector)
        
        return token_sequence,scale_sequence,chord_tone_sequence,guide_vector_sequence,tensions_vector_sequence,available_tensions_sequence,avoid_sequence
    

    def interval_to_vectors(self,intervals):
        vector = [0]*12
        for i in intervals:
            if 0<= i<12:
                vector[i]=1

        return vector



    # =================================================
    # Dataset size
    # =================================================

    def __len__(self):
       return len(self.token_data)

    # =================================================
    # Padding
    # =================================================

    def __getitem__(self,index):
        tokens = self.token_data[index]
        tokens,labels = self.mask_sequence(tokens)
        scales = self.scale_data[index]
        chord_tones = self.chord_tone_data[index]
        tensions = self.tension_data[index]
        available_tensions = self.available_tension_data[index]
        avoids = self.avoid_data[index]
        length = len(tokens)
        input_ids=torch.zeros(self.seq_length,13,dtype = torch.long)
        scale_vector = torch.zeros(self.seq_length,12,dtype=torch.float)
        chord_tones_vector = torch.zeros(self.seq_length,12,dtype=torch.float)
        guide_tones_vector = torch.zeros(self.seq_length,12,dtype=torch.float)
        tensions_vector = torch.zeros(self.seq_length,12,dtype=torch.float)
        available_tensions_vector = torch.zeros(self.seq_length,12,dtype=torch.float)
        avoid_vector = torch.zeros(self.seq_length,12,dtype=torch.float)
        
        attention_mask=torch.zeros(self.seq_length, dtype=torch.bool )
        target_ids = torch.full((self.seq_length,),-100,dtype=torch.long)

        if length > self.seq_length:
            length = self.seq_length
            tokens = tokens[:length]
            labels = labels[:length]
            scales = scales[:length]
            chord_tones = chord_tones[:length]
            tensions = tensions[:length]
            available_tensions = available_tensions[:length]
            avoids = avoids[:length]
 


        input_ids[:length]=torch.tensor(tokens,dtype=torch.long)
        target_ids[:length]=torch.tensor(labels,dtype=torch.long)

        attention_mask[:length]=True
        return {
            "input_ids":input_ids,
            "scale_vector": scale_vector,
            "chord_tones_vector": chord_tones_vector,
            "guide_tones_vector":guide_tones_vector,
            "tensions_vector":tensions_vector,
            "available_tensions_vector":available_tensions_vector,
            "avoid_vector":avoid_vector,
            "attention_mask":attention_mask,
            "labels": target_ids
        }


# =================================================
# Test
# =================================================

if __name__=="__main__":
    dataset=HarmonyDataset(
        json_file=os.path.join(HARMONY_TOKEN_DIR,"ireal_harmony_form.json"),
        vocab_dir= HARMONY_VOCAB_DIR,
        seq_length=128)
    print("Dataset size:",len(dataset))
    sample=dataset[0]


    print(sample["input_ids"].shape)
    print(sample["scale_vector"].shape)
    print(sample["chord_tones_vector"].shape)
    print(sample["guide_tones_vector"].shape)
    print(sample["tensions_vector"].shape)
    print(sample["available_tensions_vector"].shape)
    print(sample["avoid_vector"].shape)
    