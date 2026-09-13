import sys

sys.path.append(
    "/Volumes/My Passport/python_packages_AIMusic"
)
import json
from pathlib import Path
import pyRealParser
from pyRealParser import Tune
from jazz_features import JazzFeatures
import re



class IrealChordDecoder:
    def __init__(self):
        self.feature = JazzFeatures()


    def clean_chord_symbol(self,chord):
        if chord is None:
            return None
        chord = chord.strip()
        if chord == "":
            return None
        if chord.startswith("N"):
            return None
        if chord== "U":
            return None
        chord = chord.replace("U","")
        chord = chord.replace("N1","")
        chord = chord.replace("N2","")
        if chord == "":
            return None
        return chord
    # =========================
    # Chord embedding
    # =========================
    def chord_embedding(self,chord,key):
        key_id = self.feature.Note_MAP.get(key,0)
        normalized = self.feature.normalize_ireal_chord(chord)

        root = normalized["root"]
        quality = normalized["quality"]
        bass = normalized["bass"]
        normalized_chord = root + quality
        if bass and bass != root:
           normalized_chord += "/" + bass 
        embedding = self.feature.chord_embedding(normalized_chord,key_id)

        #slash chord bass analysis
        chord_tones = embedding.get("chord_tones",[])
        root_pc = self.feature.Note_MAP.get(root,0)
        bass_pc = self.feature.Note_MAP.get(bass,0)
        chord_pitch_class = [(root_pc + interval) for interval in chord_tones]
        bass_interval = (bass_pc - root_pc) % 12
        inversion = self.feature.detect_inversion(root, bass, chord_tones)
        embedding["inversion"] = inversion 
        embedding["root_note"] = root
        embedding["bass_note"] = bass 
        embedding["bass_interval"] = bass_interval
        if bass == root or bass is None:
            embedding["is_slash_chord"] = False
            embedding["bass_type"] = "root"
        elif bass_pc in chord_pitch_class :
            embedding["is_slash_chord"] = True
            embedding["bass_type"] = "chord_tone"
        
        else:
            embedding["is_slash_chord"] = True
            embedding["bass_type"] = "non_chord_tone"
            
        return embedding
  

    # =========================
    # Decode single chord
    # =========================

    def decode_chord(self,chord,key,bar,beat_start = 0, duration=4):
        raw_symbol =chord
        if chord in ["", "%","x","X"]:
            return {
                "type":"REPEAT",
                "bar":bar,
                "beat_start": beat_start,
                "duration": duration
            }
        symbol = self.feature.normalize_ireal_chord(chord)
        embedding = self.chord_embedding(chord,key)
        return {
            "type":"Chord",
            "bar":bar,
            "beat_start":beat_start,
            "duration": duration,
            "symbol":symbol,
            "raw_symbol":raw_symbol,
            "embedding":embedding
        }




    # =========================
    # Decode measures
    # =========================

    def decode_measures(self,measures,key):
        events=[]
        previous_bass = None
        for bar,measure in enumerate(measures,start=1):
            items = self.split_ireal_measure(measure)
            for item in items:
                chord = item["chord"]
                chord = self.clean_chord_symbol(chord)
                if chord is None:
                    continue
                print("RAW Chord:", chord)
                if chord.endswith("/") and chord!="/":
                    if previous_bass:
                        chord =chord[:-1]+ "/" + previous_bass
                event = self.decode_chord(
                    chord,
                    key=key,
                    bar=bar,
                    beat_start=item["beat_start"],
                    duration=item["duration"]
                )
                if event["type"] == "Chord":
                    symbol = event.get("symbol")
                    if isinstance(symbol,dict):
                        bass = symbol.get("bass")
                        if bass:
                            previous_bass = bass
                events.append(event)

        return events

    def split_ireal_measure(self,measure,beats_per_bar=4):
        if measure is None:
            return []
        measure = measure.strip()
        chords = []
        current = ""
        i = 0
        while i < len(measure):
            c = measure[i]
            if(c in "ABCDEFG" and current):
                if current.endswith("/"):
                    current += c
                else:
                    chords.append(current)
                    current = c
            else:
                current += c
            i+=1
        if current:
            chords.append(current)
        duration = beats_per_bar / len(chords)

        result = []
        current_beat = 0
        for chord in chords:
            result.append(
            {
                "chord":chord,
                "beat_start":current_beat,
                "duration":duration
            }
        )
            current_beat+=duration
        return result

    # detcet the chord progression



    # =========================
    # Process
    # =========================

    def process(self,input_file,output_file):
        with open(input_file, "r",encoding="utf-8") as f:
            songs=json.load(f)
        results=[]
        for song in songs:
            url=song["ireal_url"]
            if url is None:
                print("Missing url",song)
                continue
            key=song.get("harmony",{}).get("key","C")
            # pyRealParser
            tunes=Tune.parse_ireal_url(url)
            if not tunes:
                print("Skip invalid:", song.get("title"))
                continue
            if isinstance(tunes, list):
                tune = tunes[0]
            else: 
                tune = tunes

            chord_string = tune.chord_string
            measures = tune._get_measures(chord_string)
            # print("Prased:",song["song_info"]["title"])
            # print("Measeures:",measures[:3])
            events=self.decode_measures(measures,key)
            results.append({
                "title":song.get("song_info",{}).get("title"),
                "composer":song.get("song_info",{}).get("composer"),
                "style":song.get("style",{}).get("feel"),
                "key":key,
                "measures":measures,
                "events":events
            })

        Path(output_file).parent.mkdir(parents=True,exist_ok=True)
        with open(output_file,"w",encoding="utf-8") as f:
            json.dump(results,f,indent=4,ensure_ascii=False)
        print("Processed:",len(results))

if __name__=="__main__":
    decoder=IrealChordDecoder()
    decoder.process(
        "/Volumes/My Passport/Jazz Gen/data/processed/ireal_json/jazz1400.json",
        "/Volumes/My Passport/Jazz Gen/data/processed/ireal_harmony/ireal_harmony_embedding.json"
    )