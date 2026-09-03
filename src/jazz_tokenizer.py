import json
from pathlib import Path
from collections import Counter

Json_V1_Dir= '/Volumes/My Passport/Jazz Gen/data/processed/Jazz_json/WJazzD_JSON'
Json_V2_Dir = '/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2'
Token_OutPut_Dir = '/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/token'
Vocabulary_Output_Dir='/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/vocabulary'

class JazzTokenizer:
    def __init__(self):

        self.special_tokens=[
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>"
        ]
        self.token_to_id={}
        self.id_to_token={}


    # chord token
    def chord_tokens(self,chord):
        tokens=[]
        if chord:
            tokens.append(f"CHORD_{chord}")
        return tokens


    # note token
  
    def note_tokens(self,note):
        tokens=[]
        pitch=note["pitch"]
        time=note.get("time",0)
        duration=note.get("duration",120)
        velocity=note.get("velocity",80)

        # bar
        bar=time//1920
        beat=bar//480
        tokens.append( "<BAR>")
        tokens.append(f"BEAT_{beat}")
        # position
        position=(time%480)//60
        tokens.append(f"POSITION_{position}")

        # pitch
        tokens.append(f"PITCH_{pitch}")
        # interval
        if "previous_pitch" in note:
            interval=(pitch-note["previous_pitch"])
            tokens.append(f"INTERVAL_{interval}")
        # duration
        tokens.append(f"DURATION_{duration}")
        # velocity
        vel=velocity//10*10
        tokens.append(f"VELOCITY_{vel}")
        return tokens

    # metadata token
    def metadata_tokens(self,metadata):
        tokens=[]
        if not metadata:
            return tokens
        for key,value in metadata.items():
            if key=="style_embedding":
                continue
            tokens.append(f"{key.upper()}_{value}")
        return tokens


    # 单曲编码
 
    def encode_song(self,data):
        tokens=[]
        tokens.append("<BOS>")

        # metadata
        tokens.extend(self.metadata_tokens(data.get("metadata_v2",{})))
        # chords
        chord_map={c["time"]:c["chord"] for c in data.get("chords",[])}
        previous_pitch=None
        # melody
        for note in data["melody"]:
            time=note.get("time",0)
            if time in chord_map:
                tokens.extend(
                    self.chord_tokens(chord_map[time]))
            if previous_pitch is not None:
                note["previous_pitch"]=previous_pitch
            tokens.extend(self.note_tokens(note))
            previous_pitch=note["pitch"]
        tokens.append("<EOS>")
        return tokens


    # 建立词表
    def build_vocab(self,json_files):
        counter=Counter()
        for file in json_files:
            with open(file,encoding="utf8") as f:
                data=json.load(f)
            tokens=self.encode_song(data)
            counter.update(tokens)
        vocab=self.special_tokens.copy()
        for token,count in counter.items():
            if (count>=2 or token.startswith(("TEMPO","SWING","KEY","MODE","SCALE","STYLE"))):
                vocab.append(token)
        self.token_to_id={token:i for i,token in enumerate(vocab)}
        self.id_to_token={i:t for t,i in self.token_to_id.items()}
        print("TEMPO tokens:",sum(1 for t in vocab if t.startswith("TEMPO")))
        print("SWING tokens:",sum(1 for t in vocab if t.startswith("SWING")))
        print("Vocabulary size:",len(vocab))

    # token转id
    def convert_ids(self,tokens):
        return [self.token_to_id.get(token,self.token_to_id["<UNK>"])for token in tokens]
    
    # 保存
    def tokenize_dataset(self,input_dir,output_dir):
        input_dir=Path(input_dir)
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        for file in input_dir.glob("*.json"):
            with open(file,encoding="utf8") as f:
                data=json.load(f)
            tokens=self.encode_song(data)
            ids=self.convert_ids(tokens)
            with open(output_dir/file.name,"w",encoding="utf8") as f:
                json.dump(
                    {
                    "tokens":tokens,
                    "ids":ids},f,indent=2,ensure_ascii=False)

    def save_vocab(self,path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        vocabulary_file = path / "vocabulary.json"
        with open(vocabulary_file,"w",encoding="utf8") as f:
            json.dump({
                "token_to_id":self.token_to_id,
                "id_to_token":self.id_to_token},f,indent=2,ensure_ascii=False)
        print("Vocabulary_Saved", vocabulary_file)
if __name__=="__main__":
    tokenizer=JazzTokenizer()
    json_v2_dir = Path(Json_V2_Dir)
    files=list(json_v2_dir.glob("*.json"))
    print("Songs:",len(files))
    tokenizer.build_vocab(files)
    tokenizer.save_vocab(Vocabulary_Output_Dir)
    tokenizer.tokenize_dataset(Json_V2_Dir,Token_OutPut_Dir)
    print("Tokenizer finished")