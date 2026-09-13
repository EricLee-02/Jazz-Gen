import json
from pathlib import Path



class HarmonyProgressionAnalyzer:
    def __init__(self):
        # jazz grammar patterns
       self.patterns = {
    # =====================
    # Cadence  终止式
    # =====================
    "II-V-I":{
        "pattern":["II","V","I"
        ],
        "priority":10,
        "category":"cadence"
        },
    "Minor_II-V-I":{
        "pattern":["IIø","V","I"],
        "priority":10,
        "category":"cadence"
        },
    # =====================
    # Turnaround 
    # =====================

    "Turnaround":{
        "pattern":["I","VI","II","V"],
        "priority":8,
        "category":"form"
        },
    # =====================
    # Dominant chain 属功能连续
    # =====================
    "Dominant_Chain":{
        "pattern":["V","V","V"],
        "priority":6,
        "category":"dominant_motion"
        },
    # V of V chain
    "Extended_Dominant_Chain":{
        "pattern":["V/V","V","I"],
        "priority":9,
        "category":"tonicization" #重属和弦连接
        },
    # =====================
    # Continuous II-V
    # =====================
    "Continuous_25":{
        "pattern":["II","V","II","V","I"],
        "priority":10,
        "category":"cadence"
        },
    # =====================
    # Diminished approach 经过和先
    # =====================
    "Dim7_Approach":{
        "pattern":["dim7","I"],
        "priority":7,
        "category":"approach"
        },
    "Dim7_Passing":{
        "pattern":["I","Dim7","II"],
        "priority":6,
        "category":"approach"
        },
    # =====================
    # Tritone substitution 将二代五
    # =====================
    "Tritone_Substitution":{
        "pattern":["bII7","I"],
        "priority":8,
        "category":"chromatic_motion"
        },
    # =====================
    # Backdoor
    # =====================
    "Backdoor":{
        "pattern":["IV","bVII","I"],
        "priority":8,
        "category":"cadence"
        },
    # =====================
    # Chromatic dominant 半音/替代 运动
    # =====================
    "Chromatic_Dominant":{
        "pattern":["V","bV","I"],
        "priority":5,
        "category":"chromatic_motion"
        }
        }

       self.detectors =[
            self.detect_fixed_patterns,
            self.detect_dominant_chain,
            self.detect_dim7,
            self.detect_secondary_dominant,
            self.detect_tritone_substitution
       ]
    # ==========================
    # extract harmony sequence
    # ==========================

    def extract_harmony(self,events):
        sequence=[]
        for event in events:
            if event.get("type") != "Chord":
                continue
            embedding = event.get( "embedding",{})
            roman = embedding.get("roman_numeral")
            if roman is None:
                continue
            sequence.append({
                "bar":event.get("bar"),
                "beat_start":event.get("beat_start",0),
                "duration":event.get("duration",4),
                "roman":self.normalize_roman(roman),
                "symbol":event.get("raw_symbol"),
                "embedding":embedding
            })
        return sequence
    # ==========================
    # normalize roman
    # ==========================
    def normalize_roman(self,roman):
        if roman is None:
            return None
        roman=roman.strip()
        # remove extensions
        roman=roman.replace("7","")
        roman=roman.replace("Maj", "")
        return roman
    

    def detect_fixed_patterns(self, sequence):

        results=[]
        romans=[x["roman"] for x in sequence]
        for name,info in self.patterns.items():

            pattern=info["pattern"]

            length=len(pattern)


            for i in range(len(romans)-length+1):

               window=romans[i:i+length]
               if window == pattern:

                results.append({
                    "type":name,
                    "category":info["category"],
                    "priority":info["priority"],
                    "confidence":1.0,
                    "start_bar":sequence[i]["bar"],
                    "end_bar":sequence[i+length-1]["bar"],
                    "chords":[sequence[j]["symbol"]for j in range(i,i+length)],
                    "roman":window
                })
        return results
    
    def detect_dominant_chain(self,sequence):
        results = []
        chain = []
        for item in sequence:
            function = item["embedding"].get("function")
            if function == "Dominant":
                chain.append(item)
            else:
                if len(chain) >= 2:
                    results.append({
                        "type": "dominant_chain",
                        "category": "dominant_motion",
                        "priority": 6,
                        "confidence": 0.8,
                        "start_bar":chain[0]["bar"],
                        "end_bar": chain[-1]["bar"],
                        "chords":[x["symbol"] for x in chain]
                    })
                chain = []
        return results
    
    def detect_tritone_substitution(self,sequence):

        results=[]
        for i in range(len(sequence)-1):
            current=sequence[i]
            next_chord=sequence[i+1]
            function=current["embedding"].get("function")
            if function!="Dominant":
                continue
            root=current["embedding"].get("root_pitch_class")
            next_root=next_chord["embedding"].get("root_pitch_class")
            if root is None or next_root is None:
                continue
            interval=(next_root-root)%12
        # dominant下降半音解决
            if interval==1:
               results.append({
                "type":"Tritone_Substitution",
                "category":"chromatic_motion",
                "priority":9,
                "confidence":0.75,
                "start_bar":current["bar"],
                "end_bar":next_chord["bar"],
                "chords":[current["symbol"],next_chord["symbol"]]
                })
        return results
    

    def detect_secondary_dominant(self,sequence):
        results=[]
        for i in range(len(sequence)-1):
            current=sequence[i]
            next_chord=sequence[i+1]
            current_function=current["embedding"].get("function")
            next_function=next_chord["embedding"].get("function")
            if (current_function=="Dominant" and next_function=="SubDominant"):
                results.append({
                "type":"secondary_dominant",
                "category":"tonicization",
                "priority":8,
                "start_bar":current["bar"],
                "end_bar":next_chord["bar"],
                "chords":[current["symbol"],next_chord["symbol"]]
                })


        return results
    

    def detect_dim7(self,sequence):
        results=[]
        for i in range(len(sequence)-1):
           current=sequence[i]
           next_chord=sequence[i+1]
           attribute=current["embedding"].get("attribute","")
           next_function=next_chord["embedding"].get( "function","")
           if ("Dim" in attribute and next_function=="Tonic"):
               results.append({
                "type":"dim7_resolution",
                "category":"approach",
                "priority":7,
                "start_bar":current["bar"],
                "end_bar":next_chord["bar"],
                "strength":0.8
            })
        return results
    
    # def detect_dim_resolution(self,sequence):
    #     results = []
    #     for i in range(len(sequence)-2):
    #         c1 = sequence[i]["embedding"]
    #         c2 = sequence[i]["embedding"]
    #         if (c1["attribute"] == "Dim7" and c2["function"] == "Tonic"):
    #             results.append({
    #                 "type": "dim7_resolution",
    #                 "bar":  sequence[i]["bar"]
    #             })
    #         elif (c1["attribute"] == "Dim" and c2["function"] == "Tonic"):
    #             results.append({
    #                 "type": "dim_resolution",
    #                 "bar":  sequence[i]["bar"]
    #             })
    #     return results
    
    # ==========================
    # detect progression
    # ==========================
    def detect_progression(self,sequence):
        all_results=[]
        for detector in self.detectors:
            results = detector(sequence)
            all_results.extend(results)

        all_results.sort(key=lambda x:x.get("priority",0),reverse=True)

        return all_results

    # ==========================
    # phrase grouping
    # ==========================
    def build_phrases(self,sequence,max_gap=1):
        if not sequence:
            return []
        phrases=[]
        current=[sequence[0]]
        for item in sequence[1:]:
            last=current[-1]
            gap=item["bar"]-last["bar"]
            if gap<=max_gap:
                current.append(item)
            else:
                phrases.append(current)
                current=[item]
        phrases.append(current)
        return phrases
    


    # ==========================
    # process
    # ==========================
    def process(self,input_file,output_file):
        with open(input_file,"r",encoding="utf-8") as f:
            songs=json.load(f)
        results=[]
        for song in songs:
            events=song.get("events",[])
            harmony=self.extract_harmony(events)
            phrases=self.build_phrases(harmony)
            progressions=[]
            for phrase in phrases:
                detected=self.detect_progression(phrase)
                progressions.extend(detected)
            results.append({
                "title":song.get("title"),
                "composer":song.get("composer"),
                "key":song.get("key"),
                "progressions": progressions
            })

        Path(output_file).parent.mkdir(parents=True,exist_ok=True)
        with open(output_file,"w",encoding="utf-8") as f:
            json.dump(results,f,indent=4,ensure_ascii=False)
        print("Processed songs:",len(results))
if __name__=="__main__":
    analyzer=HarmonyProgressionAnalyzer()
    analyzer.process(
        "data/processed/ireal_harmony/ireal_harmony_embedding.json",
        "data/processed/ireal_harmony/harmony_progression.json"
    )