import numpy as np 
import re
# 0 -> Root 4-> 3th 7-> 5th 11->7th 
# plusing 1 means sharping half note, subtract 1 means flatting half note


class JazzFeatures:
    Source_Map ={
        "Ireal_Pro": 0,
        "WJazzD": 1,
        "Generated": 2
    }

    Perform_Style_Map = {
    "Swing":0,
    "Bebop":1,
    "Cool_Jazz":2,
    "Hard_Bop":3,
    "Modal_Jazz":4,
    "Post_Bop":5,
    "Free_Jazz":6,
    "Jazz_Fusion":7,
    "Latin_Jazz":8,
    "Contemporary_Jazz":9, 
    "Other":10
    }


    Ireal_Pro_Style_MAP = {
    "Medium Swing":0,
    "Slow Swing":1,
    "Uptempo Swing":2,
    "Ballad":3,
    "Bossa Nova":4,
    "Latin":5,
    "Waltz":6,
    }

    Note_MAP={
        "C":0,
        "C#":1,
        "Db":1,
        "D":2,
        "D#":3,
        "Eb":3,
        "E":4,
        "F":5,
        "F#":6,
        "Gb":6,
        "G":7,
        "G#":8,
        "Ab":8,
        "A":9,
        "A#":10,
        "Bb":10,
        "B":11
        }

    Rhythm_Map = {
        "Swing":0,
        "Straight":1,
        "Charleston":2,
        "Reverse_Charleston":3,
        "Push":4,
        "Lay_Back":5,
        "Latin":6,
        "Waltz":7,
        "Free":8
    }

    Phrase_Map ={
        "Call" : 0,
        "Respone": 1,
        "Motif" : 2,
        "Sequence" :3,
        "Climax": 4,
        "Ending": 5
    }

    Chord_Attribute_MAP = {
    "Maj":0,
    "Maj7":1,
    "Maj7#11":2,
    "6":3,
    "Min":4,
    "Min7":5,
    "MinMaj7":6,
    "Min7b5":7,
    "7":8,
    "7b9":9,
    "7#9":10,
    "7alt":11,
    "7#11":12,
    "13":13,
    "7sus4":14,
    "Dim":15,
    "Dim7":16,
    "Aug":17,
    "h7":18,
    "h":19,
    "N.C":20
    }

    Level_MAP={
        0:"I",
        1:"bII",
        2:"II",
        3:"bIII",
        4:"III",
        5:"IV",
        6:"#IV",
        7:"V",
        8:"bVI",
        9:"VI",
        10:"bVII",
        11:"VII"
               }
    ROMAN_REVERSE_MAP = {value : key for key, value in Level_MAP.items()}
    
    Cadence_MAP={
        "None":0,
        "II-V":1,
        "II_V_I":2,
        "Substitute":3,
        "Dominant_chain":4,
        "Turnaround":5,
        "Backdoor":6,
        "Modal": 7,
        "Chromatic":8
        }
    
    Chord_Function_Map = { #和弦的作用
    "Tonic":0, #I, III, VI
    "SubDominant":1,# IV, II, VI
    "Dominant":2, # V, III, VII
}

    Scale_Map = {
    "Ionian":0,
    "Dorian":1,
    "Phrygian":2,
    "Lydian":3,
    "Mixolydian":4,
    "Aeolian":5,
    "Locrian":6,

    "Melodic_Minor":7,
    "Harmonic_Minor":8,

    "Altered":9,
    "Lydian_Dominant":10,
    "Mixolydian_b9_b13":11,

    "Half_Whole_Diminished":12,
    "Whole_Tone":13,

    "Major_Bebop":14,
    "Dominant_Bebop":15,
    "Minor_Bebop":16,

    "Blues":17,
    "Minor_Pentatonic":18,
    "Major_Pentatonic":19,

    "None":20
}
    Scale_Interval_Map={
        "Ionian": [ 0,2,4,5,7,9,11],
        "Dorian": [ 0,2,3,5,7,9,10],
        "Phrygian": [0,1,3,5,7,8,10],
        "Lydian":[0,2,4,6,7,9,11],
        "Mixolydian":[ 0,2,4,5,7,9,10],
        "Aeolian":[0,2,3,5,7,8,10],
        "Locrian":[0,1,3,5,6,8,10],

        "Melodic_Minor":[0,2,3,5,7,9,11],
        "Harmonic_Minor":[0,2,3,5,7,8,11],

        "Altered":[0,1,3,4,6,8,10],
        "Lydian_Dominant":[0,2,4,6,7,9,10],
        "Mixolydian_b9_b13":[0,1,4,5,7,8,10],

        "Half_Whole_Diminished":[ 0,1,3,4,6,7,9,10],
        "Whole_Tone":[ 0,2,4,6,8,10],

        "Major_Bebop":[0,2,4,5,7,8,9,11],
        "Dominant_Bebop":[0,2,4,5,7,9,10,11],
        "Minor_Bebop":[0,2,3,4,5,7,9,10],

        "Blues":[0,3,5,6,7,10],
        "Minor_Pentatonic":[0,3,5,7,10],
        "Major_Pentatonic":[0,2,4,7,9],

        "None":[]

    }

    Chord_Alias={
    "-":"Min",
    "-7":"Min7",
    "-maj7":"MinMaj7",
    "-7b5":"Min7b5",
    "o7":"Dim7",
    "o":"Dim",
    "+":"Aug",
    "h7":"Min7b5",
    "h": "Dim"
    }
    
    Chord_Map = {
# Major7 Chord
    "Maj7":{
    "Chord_Tones":[0,4,7,11],
    "Tensions":[2,6,9],
    "Available_Tensions":[2,9],
    "Avoid":[5],
    "Scale":"Ionian",
    },
    "Maj7#11":{
    "Chord_Tones":[0,4,7,11],
    "Tensions":[2,6,9],
    "Available_Tensions":[6,2,9],
    "Avoid":[],
    "Scale":"Lydian",
    },
    "Maj":{
    "Chord_Tones":[0,4,7],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,5,9],
    "Avoid":[11],
    "Scale":"Ionian",
    },
    "6":{
    "Chord_Tones":[0,4,7,9],
    "Tensions":[2,5,9],
    "Available_Tensions":[9,5],
    "Avoid":[11],
    "Scale":"Ionian",
    },
# Minor chord

    "Min7":{
    "Chord_Tones":[0,3,7,10],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,5,9],
    "Avoid":[],
    "Scale":"Dorian",
    },
    "Min":{
    "Chord_Tones":[0,3,7],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,5,9],
    "Avoid":[11],
    "Scale":"Dorian",
    },
    "MinMaj7":{
    "Chord_Tones":[0,3,7,11],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,9],
    "Avoid":[5],
    "Scale":"Melodic_Minor",
    },
    "Min7b5":{
    "Chord_Tones":[0,3,6,10],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,5,9],
    "Avoid":[],
    "Scale":"Locrian",
    },
# Dominant Family
    "7":{
    "Chord_Tones":[0,4,7,10],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,5,9],
    "Avoid":[11],
    "Scale":"Mixolydian",
    },
    "7sus4":{
    "Chord_Tones":[0,5,7,10],
    "Tensions":[2,9],
    "Available_Tensions":[2,9],
    "Avoid":[4],
    "Scale":"Mixolydian",
    },
    "7b9":{
    "Chord_Tones":[0,4,7,10],
    "Tensions":[1,3,8],
    "Available_Tensions":[1,3,8],
    "Avoid":[],
    "Scale":"Harmonic_Minor",
    },
    "7#9":{
    "Chord_Tones":[0,4,7,10],
    "Tensions":[3,6,8],
    "Available_Tensions":[3,8],
    "Avoid":[],
    "Scale":"Altered",
    },
    "7alt":{
    "Chord_Tones":[0,4,10],
    "Tensions":[1,3,6,8],
    "Available_Tensions":[1,3,6,8],
    "Avoid":[],
    "Scale":"Altered",
    },
    "7#11":{
    "Chord_Tones":[0,4,7,10],
    "Tensions":[2,6,9],
    "Available_Tensions":[6,9],
    "Avoid":[],
    "Scale":"Lydian_Dominant"
    },
    "13":{
    "Chord_Tones":[0,4,7,10],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,9,5],
    "Avoid":[],
    "Scale":"Mixolydian",
   },

# Diminished chord
    "Dim7":{
    "Chord_Tones":[0,3,6,9],
    "Tensions":[1,4,7,10],
    "Available_Tensions":[1,4,7,10],
    "Avoid":[],
    "Scale": "Half_Whole_Diminished",
    },
    "Dim":{
    "Chord_Tones":[0,3,6],
    "Tensions":[1,4,7,10],
    "Available_Tensions":[1,4,7,10],
    "Avoid":[],
    "Scale": "Half_Whole_Diminished",
    },

# Half Diminished Chord
    "Half_Dim":{
    "Chord_Tones":[0,3,6,10],
    "Tensions":[2,5,9],
    "Available_Tensions":[2,5,9],
    "Avoid":[],
    "Scale":"Locrian",
    },
    "Aug":{
    "Chord_Tones":[0,4,8],
    "Tensions":[2,6,9],
    "Available_Tensions":[2,6,9],
    "Avoid":[],
    "Scale":"Whole_Tone",
    },
    "N.C":{
        "Chord_Tones": [],
        "guide_tones": [],
        "Tensions": [],
        "Available_Tensions":[],
        "Avoid":[],
        "Scale":"None",
    }
    }
    
    def __init__(self):
        self.normalize_chord_map()
    
    #------------------------------------------------------------------

    def style_embedding(self,style):
        if style is None:
            return self.Perform_Style_Map["Other"]
        for key,value in self.Perform_Style_Map.items():
            if key in style:
                return value
        return self.Perform_Style_Map["OTHER"]

    def ireal_pro_style(self,style):
        if style is None:
            return None
        
        style = style.strip()

        if style in self.Ireal_Pro_Style_MAP:
          return self.Ireal_Pro_Style_MAP[style]
        
        clean_style = " ".join(style.split())
        if clean_style in self.Ireal_Pro_Style_MAP:
            return self.Ireal_Pro_Style_MAP[clean_style]
        
        if "Swing" in style :
            if ("Up" in style or "Uptempo" in style):
                return self.Ireal_Pro_Style_MAP.get("Uptempo Swing")
            if "Slow" in style:
                return self.Ireal_Pro_Style_MAP.get("Slow Swing")
            return self.Ireal_Pro_Style_MAP.get("Medium Swing")
        if "Bossa" in style:
            return self.Ireal_Pro_Style_MAP.get("Bossa Nova")
        if "Latin" in style:
            return self.Ireal_Pro_Style_MAP.get("Latin")
        if "Ballad" in style:
            return self.Ireal_Pro_Style_MAP.get("Ballad")
        if "Waltz" in style:
            return self.Ireal_Pro_Style_MAP.get("Waltz")
        return self.Ireal_Pro_Style_MAP.get("Unknown",None)
    

    
 #------------------------------------------------------------------

    # velocity
    def add_velocity(self, melody, velocity_map):
        for note in melody:
            pitch = int(note["pitch"])
            note["velocity"] = velocity_map.get(pitch, 80)
        return melody
    
    def articulation(self, duration, beat_length):
        ratio = duration / beat_length

        if ratio < 0.5:
            return "staccato"
        
        if ratio > 0.9:
            return "legato"
        
        else:
            return "normal"
        
    def add_articulation(self, melody):
        for note in melody:
            beat_length = 1
            note["articulation"] = self.articulation(note["duration"], beat_length)
        return melody
    
    # micro timing

    def add_micro_timing(self, melody):
        for note in melody:
            tatum = note["tatum"]
            subtatum = note["subtatum"]
            # 理论grid
            grid = (note["bar"]*4+note["beat"]+subtatum*0.25)
            real = (note["onset"])
            note["micro_timing"] = round(real-grid,4)
        return melody
    
    # construct swing feel 
    
    def calculate_swing(self,melody):
        onsets=[x["onset"]for x in melody]
        intervals=[]
        for i in range(len(onsets)-1):
            diff=onsets[i+1]-onsets[i]
            if diff>0:
                intervals.append(diff)
        if len(intervals)==0:
            return 1.0
        ratio=np.mean(intervals)/(np.median(intervals))

        return round(float(ratio),3)
    

 #------------------------------------------------------------------
    
    def generate_guide_tones(self,chord_tones):
 
    # Jazz guide tones:
    # priority:3rd + 7th
    # Major:4,11
    # Minor:3,10
    # Dominant:4,10
        if len(chord_tones)>=4:
            return [chord_tones[1],chord_tones[3]]

        elif len(chord_tones)==3:
            return [chord_tones[1],chord_tones[2]]
        else:
            return chord_tones

    
    def get_guide_tones(self,chord_tones):
        result=[]
        for note in chord_tones:
            if note in [3,4,10,11]:
                result.append(note)
        return result
    
    def root_pitch_class(self, chord):
        if chord is None:
            return 0
        chord = chord.strip()
        if chord == "":
            return 0
        chord = chord.split("/")[0]

        if len(chord) >= 2:
            root_candidate = chord[:2]
            if root_candidate in self.Note_MAP:
                return self.Note_MAP[root_candidate]
        
        return self.Note_MAP.get(chord[0],0)
       
    def bass_pitch_class(self,chord):
        if "/" not in chord:
            return self.root_pitch_class(chord)
        bass=chord.split("/")[-1]
        return self.Note_MAP.get(bass,0)
    
    def scale_to_vector(self,scale_name):
        vector = [0] * 12
        intervals = self.Scale_Interval_Map.get(scale_name,[])
        for i in intervals:
            vector [i] = 1
        return vector    
    def get_inversion(self,root,bass):
        interval=(bass-root)%12
        inversion_map={
            0:0,   # root position
            3:1,   # minor third bass
            4:1,   # major third bass
            5:2,
            7:2,
            10:3,
            11:3
            }
        return inversion_map.get(interval,-1)
    
    def roman_numeral(self,chord,key_root):
        root = self.root_pitch_class(chord)
        degree = (root-key_root) % 12
        return self.Level_MAP.get(degree,"Unknown")
    
    def cadence_type(self,previous,current,next_chord=None):
        if previous is None:
            return "None"
        prev_degree = previous.get("roman_numeral")
        curr_degree = current.get("roman_numeral")
        next_degree = None
        if next_chord is not None:
            next_degree = next_chord.get("roman_numeral")
    # ii - V - I
        if (prev_degree=="II" and curr_degree=="V" and next_degree=="I"):
           return "II_V_I"

    # ii - V
        if (prev_degree=="II" and curr_degree=="V"):
            return "II-V"

    # Dominant chain
    # V/V/V
        if ( prev_degree=="V" and curr_degree=="V"):

            return "Dominant_chain"

    # Turnaround
    # I-vi-ii-V
        if (prev_degree=="I" and curr_degree=="VI" and next_degree=="II"):
            return "Turnaround"

    # Backdoor
    # iv-bVII-I
        if (prev_degree=="IV" and curr_degree=="bVII" and next_degree=="I"):
            return "Backdoor"
    # Modal
    # same chord repeated

        if (previous["roman_numeral"]==current["roman_numeral"] and previous["attribute"]==current["attribute"]):
            return "Modal"

    # Chromatic
        motion=(current["root_pitch_class"]-previous["root_pitch_class"])%12
        if motion in [1,11]:
            return "Chromatic"
        
        return "None"
    
    def add_cadence_embedding(self,chord_features):
        result = []
        length = len(chord_features) 

        for i, chord in enumerate(chord_features):
            previous = None
            current = chord
            next_chord = None

            if i > 0:
                previous = chord_features[i-1]
            if i < length-1:
                next_chord = chord_features[i+1]
            cadence = self.cadence_type(previous,current,next_chord)
            chord_output = chord.copy()
            chord_output["cadence"] = cadence
            chord_output["cadence_id"] = self.Cadence_MAP.get(cadence,0)
            result.append(chord_output)
        
        return result

    
    def parse_root(self,chord):
        if chord is None:
            return 0
        chord = chord.strip()
        
        for key in self.Note_MAP:
            if chord.startswith(key):
                return self.Note_MAP[key]
        return 0


 #------------------------------------------------------------------
    
    def scale_embedding(self,scale):
        if scale is None:
            return self.Scale_Map["None"]
        for key, value in self.Scale_Map.items():
            if key.lower() == scale.lower():
                return value
        return self.Scale_Map["None"]
    

    def detect_inversion(self,root,bass,chord_tones):
        if root is None or chord_tones is None:
            return 0
        root_pc = self.Note_MAP.get(root,0)
        bass_pc = self.Note_MAP.get(bass,0)
        if bass_pc == root_pc:
            return 0
        chord_pitch_class = [(root_pc + interval) % 12 for interval in chord_tones]
        try: 
            index = chord_pitch_class.index(bass_pc)
            return index
        except ValueError:
            return 4
    
    

    def get_chord_function(self,chord,key_root):

        chord_root = self.root_pitch_class(chord)
        degree = (chord_root - key_root) % 12
        function_map = {
            0:"Tonic",
            2:"SubDominant",
            4:"Tonic",
            5:"SubDominant",
            7:"Dominant",
            9:"Tonic",
            11:"Dominant"
            }
        return function_map.get(degree,"Tonic")
    
    def normalize_chord_map(self):
        for name,info in self.Chord_Map.items():
            if "Chord_Tones" in info and "chord_tones" not in info:
                info["chord_tones"]=info.pop("Chord_Tones")
            if "Tensions" in info:
                info["tensions"]=info.pop("Tensions")
            if "Available_Tensions" in info:
                info["available_tensions"]=info.pop("Available_Tensions")
            if "Avoid" in info:
                info["avoid"]=info.pop("Avoid")
            if "Scale" in info:
                info["scale"]=info.pop("Scale")
              
            
            
            info.setdefault("chord_tones",[])
            info.setdefault("tensions",[])
            info.setdefault("available_tensions",[])
            info.setdefault("avoid",[])
            info.setdefault("scale","None")

            if "guide_tones" not in info:
                info["guide_tones"] = self.generate_guide_tones(info["chord_tones"])



    def normalize_ireal_chord(self,chord):
        if chord is None or chord =="":
            return {
                "root": "N.C",
                "quality": "",
                "bass":"N.C"
                }
        chord = chord.strip()

        bass = None
        #slash chord
        if "/" in chord:
            main,bass_part =chord.split("/",1)
            chord = main
            bass = bass_part.strip()
            if bass =="":
                bass = None
        root_match = re.match(r"^([A-G][b#]?)(.*)",chord)
        if not root_match:
            return {
                "root": chord,
                "quality": "",
                "bass":bass if bass else chord
            }
        root = root_match.group(1)
        quality = root_match.group(2)

        #ireal notation
        if quality.startswith("^"):
            quality = "Maj7"
        elif quality.startswith("-maj7"):
            quality = "MinMaj7"
        elif quality.startswith("-7b5"):
            quality = "Min7b5"
        elif quality.startswith("-7"):
            quality = "Min7"
        elif quality.startswith("-"):
            quality = "Min"
        elif quality.startswith("o7"):
            quality = "Dim7" 
        elif quality.startswith("o"):
            quality = "Dim" 
        elif quality.startswith("+"):
            quality = "Aug" 
        elif quality.startswith ("h7"):
            quality = "Min7b5"
        elif quality.startswith("h"):
            quality = "Dim"
        elif quality == "":
            quality = "Maj" 
        
        if bass is None:
            bass = root

        return {
            "root" : root,
            "quality": quality,
            "bass": bass
        }

    
    def chord_transition_embedding(self,previous,current,next_chord = None):
        if previous is None:
            return {
                "root_motion":0,
                "attribute_change":0,
                "function_change":0,
                "cadence_type":0
                }
        root_motion=(current["root_pitch_class"]-previous["root_pitch_class"])%12
        attribute_change=(current["attribute_id"]-previous["attribute_id"])
        function_change=(current["function_id"]-previous["function_id"])
        cadence=self.cadence_type(previous,current,next_chord)
        return {
            "root_motion":root_motion,
            "attribute_change":attribute_change,
            "function_change":function_change,
            "cadence_type":self.Cadence_MAP.get(cadence,0)
            }
    

    def add_chord_embedding(self,chord_sequence, key):
        embeddings= []
        if isinstance(key,str):
            key_root = self.Note_MAP.get(key,0)
        else: key_root = key

        for chord in chord_sequence:
            feature = self.chord_embedding(chord,key_root)
            embeddings.append(feature)

        return embeddings
    

    def chord_embedding(self,chord,key_root = 0):
        original_chord = chord
        if "/" in chord:
            chord = chord.split("/")[0]
        if chord is None or chord.strip() == "":
           chord="N.C"
        chord=chord.strip()
        attribute=None
    # 先处理特殊符号
        for alias,target in sorted(self.Chord_Alias.items(),key=lambda x:len(x[0]),reverse=True):
            if alias in chord:
                attribute=target
                break
        if attribute is None:
            for key in sorted(self.Chord_Map.keys(),key=len,reverse=True):
                if key in chord:
                    attribute=key
                    break
        if attribute is None:
            attribute="Maj"
        info=self.Chord_Map.get(attribute,self.Chord_Map["Maj"])
        roman = self.roman_numeral(chord, key_root)
        function = self.get_chord_function(chord,key_root)
        return {
        "root_pitch_class": self.root_pitch_class(original_chord),
        "bass_pitch_class": self.bass_pitch_class(original_chord),
        "inversion":self.get_inversion(self.root_pitch_class(chord),self.bass_pitch_class(chord)),
        "roman_numeral":roman,
        "attribute":attribute,
        "attribute_id":self.Chord_Attribute_MAP.get(attribute,0),
        "function":function,
        "function_id":self.Chord_Function_Map.get(function,0),
        "chord_tones":info.get("chord_tones",[]),
        "guide_tones":info.get("guide_tones",[]),
        "tensions":info.get("tensions",[]),
        "available_tensions":info.get("available_tensions",[]),
        "avoid":info.get("avoid",[]),
        "scale":info.get("scale","None"),
        "scale_id":self.scale_embedding(info.get("scale")),
        "scale_vector": self.scale_to_vector(info.get("scale","None"))
        }
    


        # =====================================================
    # Unified Chord Analysis Interface
    # Used by JazzGenerationDataset
    # =====================================================

    def analyze_chord(self, chord):
        """
        Convert chord symbol into harmony encoder features

        Output:
        {
            chord_id,
            root,
            bass,
            bass_interval,
            inversion,
            attribute,
            scale_vector,
            chord_tones,
            guide_tones,
            tensions,
            available_tensions,
            avoid
        }
        """

        if chord is None or chord == "":
            chord = "N.C"
        # existing embedding function
        feature = self.chord_embedding(chord)
        # -------------------------------
        # chord id
        # -------------------------------
        attribute = feature["attribute"]
        chord_id = self.Chord_Attribute_MAP.get(attribute,self.Chord_Attribute_MAP["N.C"])
        # -------------------------------
        # root / bass
        # -------------------------------
        root = feature["root_pitch_class"]
        bass = feature["bass_pitch_class"]
        # -------------------------------
        # convert chord tones to vector
        # -------------------------------
        bass_interval = (bass - root) % 12
        if bass_interval == 0:
            inversion = 0       # root position

        elif bass_interval in [3,4]:
            inversion = 1       # third bass

        elif bass_interval in [6,7,8]:
            inversion = 2       # fifth bass

        else:
            inversion = 3       # other inversion

        attribute_id = self.Chord_Attribute_MAP.get(attribute,self.Chord_Attribute_MAP["N.C"])

        chord_tone_vector = [0]*12
        for interval in feature["chord_tones"]:
            chord_tone_vector[interval % 12] = 1

        guide_vector = [0]*12
        for interval in feature["guide_tones"]:
            guide_vector[interval % 12] = 1

        tension_vector = [0]*12
        for interval in feature["tensions"]:
            tension_vector[interval % 12] = 1

        available_vector = [0]*12
        for interval in feature["available_tensions"]:
            available_vector[interval % 12] = 1

        avoid_vector = [0]*12
        for interval in feature["avoid"]:
            avoid_vector[interval % 12] = 1


        return {
            "chord_id": chord_id,
            "root": root,
            "bass": bass,
            "bass_interval": bass_interval,
            "inversion":inversion,
            "attribute":attribute_id,
            "scale_vector":feature["scale_vector"],
            "chord_tones":chord_tone_vector,
            "guide_tones":guide_vector,
            "tensions":tension_vector,
            "available_tensions":available_vector,
            "avoid":avoid_vector
        }