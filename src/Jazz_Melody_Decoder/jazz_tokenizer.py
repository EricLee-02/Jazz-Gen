import json
from pathlib import Path
from collections import Counter


Json_V1_Dir = '/Volumes/My Passport/Jazz Gen/data/processed/Jazz_json/WJazzD_JSON'


Token_Output_Dir = '/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/token'

Vocabulary_Output_Dir = '/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token/vocabulary'


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



    # ===============================
    # duration quantization
    # ===============================

    def duration_token(self,duration):

        if duration < 0.08:
            return "DURATION_32"

        elif duration < 0.16:
            return "DURATION_16"

        elif duration < 0.30:
            return "DURATION_8"

        elif duration < 0.60:
            return "DURATION_4"

        elif duration < 1.2:
            return "DURATION_2"

        else:
            return "DURATION_LONG"



    # ===============================
    # velocity
    # ===============================

    def velocity_token(self,velocity):

        velocity=int(velocity)

        velocity=velocity//10*10

        return f"VELOCITY_{velocity}"



    # ===============================
    # micro timing
    # ===============================

    def micro_token(self,micro):

        if micro < -20:
            return "MICRO_EARLY"

        elif micro > 20:
            return "MICRO_LATE"

        else:
            return "MICRO_GRID"



    # ===============================
    # metadata
    # ===============================

    def metadata_tokens(self,metadata):
        tokens=[]

        if not metadata:
            return tokens
        keep=[
            "key",
            "mode",
            "scale",
            "style",
            "avgtempo",
            "swing_ratio"
        ]

        for k in keep:
            if k in metadata:
                value=metadata[k]

                tokens.append( f"{k.upper()}_{value}" )
        return tokens



    # ===============================
    # note token
    # ===============================

    def note_tokens(self,note):
        tokens=[]
        tokens.append(f"SECTION_{note.get('form','I1')}")
        tokens.append( f"BAR_{note.get('bar',0)}")
        tokens.append( f"PERIOD_{note.get('period',4)}")
        tokens.append(f"BEAT_{note.get('beat',0)}")
        tokens.append( f"DIVISION_{note.get('division',1)}" )
        tokens.append( f"TATUM_{note.get('tatum',0)}")
        tokens.append(f"PITCH_{int(note['pitch'])}")
        tokens.append(self.duration_token(note.get("duration",0.25)))
        tokens.append(self.velocity_token(note.get("velocity",80)))
        # if "articulation" in note:tokens.append( f"ARTIC_{note['articulation']}")
        # if "micro_timing" in note:
            # tokens.append( self.micro_token(note["micro_timing"]))
        return tokens



    # ===============================
    # chord token
    # only when chord changes
    # ===============================

    def chord_token(self,note):
        tokens =[]
        chord = note.get("chord")
        feature = note.get("chord_feature",{})
        if chord is None:
            return tokens
        if chord in ["","N.C","NC","None","nan"]:
            return tokens
        tokens.append(f"CHORD_{chord}")
        root = feature.get("root_pitch_class")
        if root is not None : 
            tokens.append(f"ROOT_PC_{root}")
        roman = feature.get("roman_numeral")
        if roman:
            tokens.append(f"ROMAN_{roman}")
        quality =feature.get("attribute")
        if quality:
            tokens.append(f"QUALITY_{quality}")
        function = feature.get("function")
        if function:
            tokens.append(f"FUNCTION_{function}")
        return tokens



    # ===============================
    # encode song
    # ===============================

    def encode_song(self,data):
        tokens=[]
        tokens.append("<BOS>")
        # metadata
        tokens.extend(self.metadata_tokens(data.get("metadata",{})))
        previous_chord=None
        for note in data["melody"]:
            current_chord=note.get("chord")
            # chord change
            if current_chord:
                current_chord = str(current_chord).strip()
            if current_chord in ["","N.C","NC","None","nan"]:
                current_chord = None
            if current_chord != previous_chord:
                if current_chord:
                    tokens.extend(self.chord_token(note))
                previous_chord = current_chord
            else:
                tokens.extend(self.chord_token(note))
            tokens.extend(self.note_tokens(note))
        tokens.append("<EOS>")
        return tokens
 




    # ===============================
    # build vocabulary
    # ===============================

    def build_vocab(self,json_files):
        counter=Counter()
        for file in json_files:
            with open(file,encoding="utf8") as f:
                data=json.load(f)
            tokens=self.encode_song(data)
            counter.update(tokens)
        vocab=self.special_tokens.copy()
        valid=[]
        for token,count in counter.items():
            if count>=2:
                valid.append(token)
        vocab.extend(sorted(valid))
        vocab=list(dict.fromkeys(vocab))
        self.token_to_id={ t:i for i,t in enumerate(vocab) }
        self.id_to_token={i:t for t,i in self.token_to_id.items()}
        print( "Vocabulary size:",len(vocab) )

    # ===============================
    # convert
    # ===============================
    def convert_ids(self,tokens):
        return [ self.token_to_id.get( t,self.token_to_id["<UNK>"]) for t in tokens]

    # ===============================
    # tokenize dataset
    # ===============================
    def tokenize_dataset( self,input_dir,output_dir):
        input_dir=Path(input_dir)
        output_dir=Path(output_dir)
        output_dir.mkdir( parents=True,exist_ok=True)
        for file in input_dir.glob("*.json"):
            with open(file,encoding="utf8") as f:
                data=json.load(f)
            tokens=self.encode_song(data)
            ids=self.convert_ids(tokens)
            with open(output_dir/file.name, "w", encoding="utf8") as f:
                json.dump( { "tokens":tokens,"ids":ids },f, indent=2,ensure_ascii=False)

    # ===============================
    # save vocab
    # ===============================

    def save_vocab(self,path):
        path=Path(path)
        path.mkdir(parents=True,exist_ok=True)
        with open(path/"vocabulary.json","w",encoding="utf8") as f:
            json.dump( {"token_to_id":self.token_to_id,"id_to_token":self.id_to_token}, f, indent=2,ensure_ascii=False)
        print("Vocabulary saved")





if __name__=="__main__":
    tokenizer=JazzTokenizer()
    files=list( Path(Json_V1_Dir).glob("*.json"))
    print( "Songs:", len(files) )
    tokenizer.build_vocab(files)
    tokenizer.save_vocab(Vocabulary_Output_Dir)
    tokenizer.tokenize_dataset(Json_V1_Dir,Token_Output_Dir)
    print("Tokenizer finished")

