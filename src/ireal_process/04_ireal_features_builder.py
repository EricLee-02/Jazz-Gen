import json
from pathlib import Path
from jazz_features import JazzFeatures


class IrealFeatureBuilder:
    def __init__(self,input_file,output_file):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.feature = JazzFeatures()
    # ---------------------------------
    # key处理
    # ---------------------------------

    def parse_key(self,key):
        if key is None:
            return 0
        key = key.strip()
        # iReal可能:
        # Bb
        # C#
        # F#m
        key = key.replace("m","")

        return self.feature.Note_MAP.get(key,0)



    # ---------------------------------
    # chord处理
    # ---------------------------------

    def process_chords(self,chords,key):
        if not chords:
            return []
        key_root=self.parse_key(key)
        # chord embedding
        chord_features = self.feature.add_chord_embedding(chords,key_root)
        # cadence embedding
        chord_features = self.feature.add_cadence_embedding(chord_features)
        return chord_features



    # ---------------------------------
    # style
    # ---------------------------------

    def process_style(self,style):

        return self.feature.ireal_pro_style(style)

    # ---------------------------------
    # 单首歌曲
    # ---------------------------------
    def process_song(self,song):
        title=song.get("song_info",{}).get("title","Unknown")
        composer=song.get("song_info",{}).get("composer","Unknown")
        style=song.get("style",{}).get("feel",None)
        key=song.get("harmony",{}).get("key","C")
        tempo=song.get("style",{}).get("tempo",None)
        raw_chord_data=song.get("harmony",{}).get("raw_chord_data",[])
        
        result={
            # metadata
            "title":title,
            "composer":composer,
            # source
            "source":
            self.feature.Source_Map["Ireal_Pro"],
            # performance
            "style":
            self.process_style(style),
            "key":key,
            "key_id":self.parse_key(key),
            "tempo":tempo,
            # harmony
            "raw_chord_data":raw_chord_data
        }
        return result



    # ---------------------------------
    # 全部处理
    # ---------------------------------

    def build(self):
        with open(self.input_file,"r",encoding="utf-8") as f:
            data=json.load(f)
        output=[]
        print("Songs:",len(data))

        for _,song in enumerate(data):
            try:
                result=self.process_song(song)
                output.append(result)
            except Exception as e:
                print("Skip:",song.get("title"),e)

        print("Processed:",len(output))

        self.output_file.parent.mkdir(parents=True,exist_ok=True)
        with open(self.output_file,"w",encoding="utf-8") as f:
            json.dump(output,f,indent=4,ensure_ascii=False)
        print("Saved:",self.output_file)

if __name__=="__main__":
    builder=IrealFeatureBuilder(
        input_file= "/Volumes/My Passport/Jazz Gen/data/processed/ireal_json/jazz1400.json",
        output_file="/Volumes/My Passport/Jazz Gen/data/processed/ireal_features/ireal_features.json"
        )
    builder.build()