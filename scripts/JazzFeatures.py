import numpy as np 

# 0 -> Root 4-> 3th 7-> 5th 11->7th 
# plusing 1 means sharping half note, subtract 1 means flatting half note


class JazzFeatures:
    STYLE_MAP={
    "BEBOP":[1,0,0,0,0],
    "HARD_BOP":[0,1,0,0,0],
    "COOL":[0,0,1,0,0],
    "FREE":[0,0,0,1,0],
    "OTHER":[0,0,0,0,1]
    }

    QUALITY_MAP = {
    "maj":0,
    "maj7":1,
    "maj7#11":2,
    "6":3,
    "min":4,
    "min7":5,
    "minMaj7":6,
    "min7b5":7,
    "7":8,
    "7b9":9,
    "7#9":10,
    "7alt":11,
    "7#11":12,
    "13":13,
    "7sus4":14,
    "dim":15,
    "dim7":16,
    "aug":17,
    "N.C":18
    }

    ROOT_MAP={
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
    
    ROMAN_MAP={
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
    
    ROMAN_REVERSE_MAP = {value : key for key, value in ROMAN_MAP.items()}

    CADENCE_MAP={
        "none":0,
        "ii-V":1,
        "authentic":2,
        "dominant_chain":3,
        "turnaround":4,
        "backdoor":5
        }

    SCALE_MAP = {
    "major_ionian":0,
    "major":1,
    "lydian":2,
    "dorian":3,
    "melodic_minor":4,
    "locrian":5,
    "mixolydian":6,
    "harmonic_minor":7,
    "altered":8,
    "lydian_dominant":9,
    "diminished":10,
    "whole_tone":11,
    "none":12
}

    FUNCTION_MAP = {
    "tonic":0,
    "minor_tonic":1,
    "ii_minor":2,
    "dominant":3,
    "passing":4,
    "none":5
}
    CHORD_ALIAS={
    "-":"min",
    "-7":"min7",
    "-maj7":"minMaj7",
    "-7b5":"min7b5",
    "o7":"dim7",
    "o":"dim",
    "+":"aug"
    }
    
    CHORD_MAP = {
# Major7 Chord
    "maj7":{
    "chord_tones":[0,4,7,11],
    "tensions":[2,6,9],
    "available_tensions":[2,9],
    "avoid":[5],
    "scale":"major_ionian",
    "function":"tonic"
    },
    "maj7#11":{
    "chord_tones":[0,4,7,11],
    "tensions":[2,6,9],
    "available_tensions":[6,2,9],
    "avoid":[],
    "scale":"lydian",
    "function":"tonic"
    },
    "maj":{
    "chord_tones":[0,4,7],
    "tensions":[2,5,9],
    "available_tensions":[2,5,9],
    "avoid":[11],
    "scale":"major",
    "function":"tonic"
    },
    "6":{
    "chord_tones":[0,4,7,9],
    "tensions":[2,5,9],
    "available_tensions":[9,5],
    "avoid":[11],
    "scale":"major",
    "function":"tonic"
    },
# Minor chord

    "min7":{
    "chord_tones":[0,3,7,10],
    "tensions":[2,5,9],
    "available_tensions":[2,5,9],
    "avoid":[],
    "scale":"dorian",
    "function":"minor_tonic"
    },
    "min":{
    "chord_tones":[0,3,7],
    "tensions":[2,5,9],
    "available_tensions":[2,5,9],
    "avoid":[11],
    "scale":"minor_dorian",
    "function":"minor_tonic"
    },
    "minMaj7":{
    "chord_tones":[0,3,7,11],
    "tensions":[2,5,9],
    "available_tensions":[2,9],
    "avoid":[5],
    "scale":"melodic_minor",
    "function":"minor_tonic"
    },
    "min7b5":{
    "chord_tones":[0,3,6,10],
    "tensions":[2,5,9],
    "available_tensions":[2,5,9],
    "avoid":[],
    "scale":"locrian",
    "function":"ii_minor"
    },
# Dominant Family
    "7":{
    "chord_tones":[0,4,7,10],
    "tensions":[2,5,9],
    "available_tensions":[2,5,9],
    "avoid":[11],
    "scale":"mixolydian",
    "function":"dominant"
    },
    "7sus4":{
    "chord_tones":[0,5,7,10],
    "tensions":[2,9],
    "available_tensions":[2,9],
    "avoid":[4],
    "scale":"mixolydian",
    "function":"dominant"
    },
    "7b9":{
    "chord_tones":[0,4,7,10],
    "tensions":[1,3,8],
    "available_tensions":[1,3,8],
    "avoid":[],
    "scale":"harmonic_minor",
    "function":"dominant"
    },
    "7#9":{
    "chord_tones":[0,4,7,10],
    "tensions":[3,6,8],
    "available_tensions":[3,8],
    "avoid":[],
    "scale":"altered",
    "function":"dominant"
    },
    "7alt":{
    "chord_tones":[0,4,10],
    "tensions":[1,3,6,8],
    "available_tensions":[1,3,6,8],
    "avoid":[],
    "scale":"altered",
    "function":"dominant"
    },
    "7#11":{
    "chord_tones":[0,4,7,10],
    "tensions":[2,6,9],
    "available_tensions":[6,9],
    "avoid":[],
    "scale":"lydian_dominant",
    "function":"dominant"
    },
    "13":{
    "chord_tones":[0,4,7,10],
    "tensions":[2,5,9],
    "available_tensions":[2,9,5],
    "avoid":[],
    "scale":"mixolydian",
   "function":"dominant"
   },

# Diminished chord
    "dim7":{
    "chord_tones":[0,3,6,9],
    "tensions":[1,4,7,10],
    "available_tensions":[1,4,7,10],
    "avoid":[],
    "scale":"diminished",
    "function":"passing"
    },
    "dim":{
    "chord_tones":[0,3,6],
    "tensions":[1,4,7,10],
    "available_tensions":[1,4,7,10],
    "avoid":[],
    "scale":"diminished",
    "function":"passing"
    },

# Half Diminished Chord
    "half_dim":{
    "chord_tones":[0,3,6,10],
    "tensions":[2,5,9],
    "available_tensions":[2,5,9],
    "avoid":[],
    "scale":"locrian",
    "function":"ii_minor"
    },
    "aug":{
    "chord_tones":[0,4,8],
    "tensions":[2,6,9],
    "available_tensions":[2,6,9],
    "avoid":[],
    "scale":"whole_tone",
    "function":"dominant"
    },
    "N.C":{
        "chord_tones": [],
        "guide_tones": [],
        "tensions": [],
        "available_tensions":[],
        "avoid":[],
        "scale":"none",
        "function":"none"
    }
    }
    
    def __init__(self):
        self.normalize_chord_map()
    
    def normalize_chord_map(self):
        for name,info in self.CHORD_MAP.items():
            if "guide_tones" not in info:
                info["guide_tones"]=self.generate_guide_tones(info["chord_tones"])


    def parse_root(self,chord):
        ROOT_MAP={
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
        for key in ROOT_MAP:
            if chord.startswith(key):
                return ROOT_MAP[key]
        return 0

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
    
    def chord_embedding(self,chord):
        if chord is None or chord.strip() == "":
           chord="N.C"
        chord=chord.strip()
        quality=None
    # 先处理特殊符号
        for alias,target in self.CHORD_ALIAS.items():
            if alias in chord:
                quality=target
                break
        if quality is None:
            for key in sorted(self.CHORD_MAP.keys(),key=len,reverse=True):
                if key in chord:
                    quality=key
                    break
        if quality is None:
            quality="maj"
        info=self.CHORD_MAP.get(quality,self.CHORD_MAP["maj"])
        return {
        "root_pitch_class": self.root_pitch_class(chord),
        "bass_pitch_class": self.bass_pitch_class(chord),
        "inversion":self.get_inversion(self.root_pitch_class(chord),self.bass_pitch_class(chord)),
        "roman_numeral":self.roman_numeral(chord),
        "quality":quality,
        "quality_id":self.QUALITY_MAP.get(quality,0),
        "chord_tones":info.get("chord_tones",[0,4,7]),
        "guide_tones":info.get("guide_tones",self.get_guide_tones(info["chord_tones"])),
        "tensions":info.get("tensions",[]),
        "available_tensions":info.get("available_tensions",[]),
        "avoid":info.get("avoid",[]),
        "scale":info.get("scale","major"),
        "scale_id":self.SCALE_MAP.get(info.get("scale"),0),
        "function":info.get("function","tonic"),
        "function_id":self.FUNCTION_MAP.get(info.get("function"),0)
        }
    
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

    def add_chord_embedding(self,melody,beats):
        previous_feature = None
        for note in melody:
            current_chord="Cmaj"
            for beat in beats:
                if (beat["onset"]<=note["onset"]):
                    current_chord=beat["chord"]
            feature = self.chord_embedding(current_chord)
            # if current_chord=="":
            #     print("Empty chord at onset:", note["onset"])
            feature["transition"] = (self.chord_transition_embedding(previous_feature,feature))
            if current_chord is None or current_chord.strip()=="":
                current_chord = "N.C"
            note["chord"]=current_chord
            note["chord_feature"]=(self.chord_embedding(current_chord))
            previous_feature = feature
        return melody
    
    def style_embedding(self,style):
        style=style.upper()
        for key,value in self.STYLE_MAP.items():
            if key in style:
                return value
        return self.STYLE_MAP["OTHER"]
    
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
        if chord =="" :
            return 0

        if len(chord) >= 2:
            root_candidate = chord[:2]
            if root_candidate in self.ROOT_MAP:
                return self.ROOT_MAP[root_candidate]
        root_candidate = chord[0]
        return self.ROOT_MAP.get(root_candidate,0)
       
    def bass_pitch_class(self,chord):
        if "/" not in chord:
            return self.root_pitch_class(chord)
        bass=chord.split("/")[-1]
        return self.ROOT_MAP.get(bass,0)
    
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
    
    def roman_numeral(self,chord,key_root=0):
        root=self.root_pitch_class(chord)
        relative=(root-key_root)%12
        return self.ROMAN_MAP.get(relative,"I")
    
    def cadence_type(self,previous,current):
        if previous is None:
            return "none"
        prev_function=previous.get("function","")
        curr_function=current.get("function", "")
        # ii-V
        if(prev_function=="ii_minor" and curr_function=="dominant"):
            return "ii-V"
        # V-I
        if (prev_function=="dominant" and curr_function=="tonic"):
            return "authentic"
         # V-V
        if (prev_function=="dominant" and curr_function=="dominant"):
            return "dominant_chain"
        return "none"
    
    def chord_transition_embedding(self,previous,current):
        if previous is None:
            return {
                "root_motion":0,
                "quality_change":0,
                "function_change":0,
                "cadence_type":0
                }
        root_motion=(current["root_pitch_class"]-previous["root_pitch_class"])%12
        quality_change=(current["quality_id"]-previous["quality_id"])
        function_change=(current["function_id"]-previous["function_id"])
        cadence=self.cadence_type(previous,current)
        return {
            "root_motion":root_motion,
            "quality_change":quality_change,
            "function_change":function_change,
            "cadence_type":self.CADENCE_MAP.get(cadence,0)
            }