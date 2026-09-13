import json
from pathlib import Path
from collections import Counter
import re 
import math

BASE_DIR = Path(__file__).resolve().parent.parent
input_dir = BASE_DIR / "data/processed/Jazz_json/WJazzD_JSON"  
output_dir = BASE_DIR / "data/processed/jazz_json_v2"
output_dir.mkdir(parents=True,exist_ok=True)
class JazzMetadataProcessor:
    def __init__(self):
        # chromatic pitch class
        self.pitch_to_class={
            "C":0,

            "C#": 1,
            "Db": 1,

            "D": 2,

            "D#": 3,
            "Eb": 3,

            "E": 4,

            "F": 5,

            "F#": 6,
            "Gb": 6,

            "G": 7,

            "G#": 8,
            "Ab": 8,

            "A": 9,

            "A#": 10,
            "Bb": 10,

            "B": 11
        }
        # Krumhansl major profile
        self.major_profile=[
            6.35,
            2.23,
            3.48,
            2.33,
            4.38,
            4.09,
            2.52,
            5.19,
            2.39,
            3.66,
            2.29,
            2.88
            ]
        self.minor_profile=[
            6.33,
            2.68,
            3.52,
            5.38,
            2.60,
            3.53,
            2.54,
            4.75,
            3.98,
            2.69,
            3.34,
            3.17
            ]

    # Key Detection
    def parse_key(self, metadata):
       if "key" not in metadata :
           return "UN","unknown"
       key_string=metadata["key"].strip()
       parts=key_string.split("-")
       key=parts[0].strip()
       mode="major"
       if len(parts)>1:
            mode_text = parts[1].lower()
            if "min" in mode_text:
                mode= "minor"
            elif "maj" in mode_text:
                mode = "major"
       return key,mode



    def parse_chords(self,chord_text):
        chords=[]
        if not chord_text:
            return chords
        chord_text = chord_text.replace("\n","")
        chord_text = chord_text.replace("A1", "")
        chord_text = chord_text.replace("A2", "")
        chord_text = chord_text.replace("A3", "")
        chord_text = chord_text.replace("B1", "")
        chord_text = chord_text.replace("B2", "")
        chord_text = chord_text.replace("B3", "")
        chord_text = chord_text.replace("C1", "")
        chord_text = chord_text.replace("C2", "")
        chord_text = chord_text.replace("C3", "")
        
        bars=chord_text.split("|")
        time=0
        for bar in bars:
            bar = bar.strip()
            if not bar:
                continue
            if ":" in bar:
                bar = bar.split(":")[-1]
            chrod_list = bar.split()
            for chord in chrod_list:
                    chord = self.normalize_chord(chord)
                    chords.append({
                        "time": time,
                        "chord": chord
                    })
                    time+=960
        return chords

    def normalize_chord(self,chord):
        chord = chord.strip()
        chord=chord.replace("-","m")
        return chord


    # Scale Detection
    def detect_scale(self,data,key):
        scale_patterns={
            "major":[0,2,4,5,7,9,11],
            "minor":[0,2,3,5,7,8,10],
            "dorian":[0,2,3,5,7,9,10],
            "mixolydian":[0,2,4,5,7,9,10]
            }
        root=self.pitch_to_class.get(key,0)
        notes=set()
        for event in data["melody"]:
            if "pitch" in event:
                interval=(event["pitch"]%12-root)%12
                notes.add(interval)
        best="major"
        score=0
        for name,pattern in scale_patterns.items():
            current=len(notes.intersection(pattern))
            if current>score:
                score=current
                best=name
        return best


    # Chord Recognition

    def detect_chords(self,data):
        chords=[]
        bars={}
        for note in data["melody"]:
            bar=note.get("bar",note.get("time",0)//1920)
            if bar not in bars:
                bars[bar]=[]
            bars[bar].append(note["pitch"]%12)
        chord_templates={
            "maj7":[0,4,7,11],
            "min7":[0,3,7,10],
            "7":[0,4,7,10],
            "dim":[0,3,6]
        }
        for bar,notes in bars.items():
            counter=Counter(notes)
            root=counter.most_common(1)[0][0]
            best="maj7"
            best_score=0
            for name,template in chord_templates.items():
                score=0
                for interval in template:
                    target=(root+interval)%12
                    if target in notes:
                        score+=1
                if score>best_score:
                    best_score=score
                    best=name
            chords.append(
                {
                "bar":bar,
                "time":bar*1920,
                "chord":self.pitch_names[root]+best
                }
            )
        return chords

    # Swing Detection
  


    # 添加bar beat

    def add_structure(self,data):
        for note in data["melody"]:
            time=note.get("time",0)
            note["bar"]=time//1920
            note["beat"]=(time%1920)//480
        return data


    # 主处理
    def process(self,input_file,output_file):
        with open(input_file,encoding="utf8") as f:
            data=json.load(f)
        data=self.add_structure(data)
        old_metadata = data.get("metadat",{})
        old_metadata = data["metadata"]
        key,mode=self.parse_key(old_metadata)
        scale=mode
        metadata={
            "key":key,
            "mode":mode,
            "scale":scale,
            "style":old_metadata.get("style","unknown").lower(),
            "style_embedding": old_metadata.get("style_embedding",[]),
            "tempo": old_metadata.get("avgtempo",120),
            "swing_ratio":old_metadata.get("swing_ratio",1),
        }
        data["metadata_v2"]=metadata
        data["chords"]=self.parse_chords(old_metadata.get("chord_changes",""))
        with open(output_file,"w",encoding="utf8") as f:
            json.dump(data,f,indent=2,ensure_ascii=False)

if __name__=="__main__":
    processor=JazzMetadataProcessor()
    files=list(input_dir.glob("*.json"))
    print("Input directory:",input_dir)

    print("JSON files:",len(files))
    for file in files:
        print("processing:",file.name)
        processor.process(file,output_dir/file.name)
    print("Jazz JSON V2 finished")