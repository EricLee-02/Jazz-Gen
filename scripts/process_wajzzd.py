import json
import sqlite3
import mido 
import pandas as pd 
import traceback
from JazzFeatures import JazzFeatures
from pathlib import Path 

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Path(__file__) : '/Volumes/My Passport/Jass Gen/scripts/process.py'
# Path(__file__).resolve() absolute path
# Path(__file__).resolve().parents[0] '/Volumes/My Passport/Jass Gen/scripts'; .parents[1] /Volumes/My Passport/Jass Gen

DB_PATH = PROJECT_ROOT/"data/raw/WJazzD/wjazzd.db"
MIDI_DIR = PROJECT_ROOT/"data/raw/WJazzD/midi"
OUTPUT_DIR = PROJECT_ROOT/"data/processed/Jazz_json/WJazzD_JSON"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Search all midi_file name
for f in sorted(MIDI_DIR.glob("*.mid")):
    print(f.stem)   



# ============================================================
# Database
# ============================================================



class Data:
    def __init__(self,db_path):
        # read the WJazzD SQLite database
        self.db_path=db_path
        if not self.db_path.exists():
            raise FileNotFoundError(f'Database not found:{self.db_path}')
        
    def get_connections(self):
        # create the sql connection
        return sqlite3.connect(self.db_path)
    
    def get_all_melids(self):
        # get all melody IDs from WJazzD
        querry='select distinct melid from solo_info order by melid'
        with self.get_connections() as conn:
            df = pd.read_sql_query(querry, conn) # get a dataframe
        return (df["melid"].dropna().astype(int).tolist()) # select the 'melid' col and drop the Nan then make it be a list
    
    def get_solo_info(self,conn,melid):
        #Read the Metadata for each solo 
        #include: Melid, Performer, title, instrument, style, avgtempo, tempoclass, rhythmfeel, key, signature, chord_progression, chorus_count
        # return a dictionary
        querry='select melid, performer, title, instrument, style, avgtempo, tempoclass, rhythmfeel, key, signature, chord_changes, chorus_count from solo_info where melid = ?' #melid = ? let melid be a paremeter
        
        df = pd.read_sql_query(querry,conn,params=(melid,))
        if df.empty: 
                raise ValueError(f'Not solo_info found for melid = {melid}')
        return df.iloc[0].to_dict()
    
    def get_transcription_info(self,conn,melid):
        #Read the Metadata for each solo 
        #include: melid, filename_solo,filename_sv, solotime, solostart_sec, status, trackid
        # return a dictionary
        querry='select melid, filename_solo, filename_sv, solotime, solostart_sec, status, trackid from transcription_info where melid = ?' #melid = ? let melid be a paremeter
        
        df = pd.read_sql_query(querry,conn,params=(melid,))
        if df.empty: 
                raise ValueError(f'Not transcription_info found for melid = {melid}')
        
        result = df.iloc[0].to_dict()
        return result

    def get_melody(self,conn,melid):
        #Read the Metadata for each solo 
        #include: eventid, onset, pitch, duration, period, division, bar, beat, tatum, subtatum, num, denom
        # return a dictionary
        querry='select eventid, onset, pitch, duration, period, division, bar, beat, tatum, subtatum, num, denom from melody where melid = ? order by onset ASC, eventid ASC' #melid = ? let melid be a paremeter
        
        df = pd.read_sql_query(querry,conn,params=(melid,))
        if df.empty: 
                raise ValueError(f'Not melody found for melid = {melid}')
        return df #make the dataframe be a dictionary and let each clo be a key related row be value

    def get_beat(self,conn,melid):
        #Read the Metadata for each solo 
        #include: beatid, onset, bar,beat, signature, chord, form, bass_pitch, chours_id
        # return a dictionary
        querry='select beatid, onset, bar,beat, signature, chord, form, bass_pitch, chorus_id from beats where melid = ? order by onset ASC, beatid ASC' #melid = ? let melid be a paremeter
        
        df = pd.read_sql_query(querry,conn, params=(melid,))
        if df.empty: 
                raise ValueError(f'Not beats found for melid = {melid}')
        return df
    
    def process_one(self,melid):
        #Read all database info for each solo
        with self.get_connections() as conn:
            metadata = self.get_solo_info(conn,melid)
            transcription = self.get_transcription_info(conn,melid)
            melody = self.get_melody(conn, melid)
            beats = self.get_beat(conn, melid)
        
        return {
            "metadata": metadata,
            "transcription": transcription,
            "melody": melody ,
            "beats": beats
        }

    

class Midi:
    # find and read midi files
    def __init__(self, midi_dir):
        self.midi_dir = midi_dir
        self.midi_index={}
        self._build_midi_index()
        if not self.midi_dir.exists():
            raise FileNotFoundError(f'MidiFile Not found {self.midi_dir} ')
    
    def _build_midi_index(self):
        for midi_file in self.midi_dir.glob("*.mid"):
            key=(
                midi_file.stem.lower()
                .replace("_final","")
                .replace("_","")
                .replace("-","")
                .replace("=","")
                .replace(" ","")
            )
            self.midi_index[key] = midi_file
        
        print(f"Loaded MIDI index: {len(self.midi_index)} files")

    def _find_midi(self, filename_solo):

        key = (filename_solo
               .lower()
               .replace("_solo","")
               .replace("_","")
               .replace("-","")
               .replace("=","")
               .replace(" ",""))
        aliases = {
        # Fats Navarro
        "fatsnavarrogoodbaitno1":
        "fatsnavarrogoodbait",
        "fatsnavarrogoodbaitno2":
        "fatsnavarrogoodbait",
        "sonnyrollinsi'llrememberaprilalternatetake2":
        "sonnyrollinsi'llrememberapril",
        }
        key = aliases.get(key, key)

        if key in self.midi_index:
            midi_path = self.midi_index[key]
            print(f"Found MIDI: {midi_path.name}")
            return midi_path

        raise FileNotFoundError(f"Midi File Not Found {filename_solo}")
    
    def inspect_midi(self, midi_path):
        #Read basic Midi information and convert the info into dict

        midi = mido.MidiFile(midi_path)

        tracks=[]

        for track_id, track in enumerate(midi.tracks):
            tracks.append({"track_index": track_id,
                           "num_messages": len(track)})
        return {'filename': midi_path.name, 
                'ticks_per_beat': midi.ticks_per_beat,
                'num_tracks': len(midi.tracks),
                'tracks': tracks
                }
    
    def extract_velocity(self,midi_path):

        midi=mido.MidiFile(midi_path)
        velocities={}
        current_time=0
        for track in midi.tracks:
            current_time=0
            for msg in track:
                current_time += msg.time
                if msg.type=="note_on" and msg.velocity>0:
                    velocities[msg.note]=msg.velocity
        return velocities
    
    def process_midi(self, filename_solo):
        # find and inspect Midi file
        midi_path = self._find_midi(filename_solo)
        midi_info = self.inspect_midi(midi_path)
        midi_info["filename"] = midi_path.name
        midi_info["path"] = midi_path

        return midi_info




class Preprocess:
    # Combine WJazzD database data and Midi data, align melody events with chord/beat events, and prepare JSON samples.
    def __init__(self, data, midi, output_dir):
        self.data = data 
        self.midi = midi 
        self.output_dir = output_dir
        self.features = JazzFeatures()
        self.output_dir.mkdir(parents = True, exist_ok = True)
    
    def align(self, melody_df, beats_df):
        # Align each melody event with the beat/chord which is active at the onset
        #Examples 
        # Melody: onset = 1.2
        # Beats: onset = 1 -> CMaj7 onset = 2 -> Dmin7
        # Aliign: onset = 1.2 -> CMaj7
        # The previous beat/chord is used untile the next beat/chord start

        melody = melody_df.copy()
        beats = beats_df.copy()

        melody = melody.sort_values(["onset", "eventid"]).reset_index(drop = True)
        beats = beats.sort_values(["onset", "beatid"]).reset_index(drop = True)

        # keep the info from beats we want to attach to every melody event

        beats_for_merge = beats[["onset", "bar", "beat", "signature", "chord", "form", "bass_pitch", "chorus_id"]].copy()

        beats_for_merge = beats_for_merge.rename(
            columns = {
                "onset": "beat_onset"
            }
        )

        aligned = pd.merge_asof(
            melody,
            beats_for_merge,
            left_on = 'onset',
            right_on = 'beat_onset',
            direction= "backward"
        ) #base on the onset and find the value in beat_onset which is <= onset in order to combine the melody with beat

        aligned["delta_time"]=(aligned["onset"]-aligned["onset"].shift(1))
        aligned["delta_time"]=(aligned["delta_time"].fillna(0).round(3))
        aligned["rest"]=(aligned["delta_time"]-aligned["duration"])
        aligned["rest"]=(aligned["rest"].clip(lower=0).round(3))
        aligned["position"]=(aligned["beat_x"].astype(str)+ "-" + aligned["tatum"].astype(str))

        #calculate position inside current chord/beat
        # example: 
        # onset = 1
        # beat_onset = 1.5
        # chord = CMaj7 
        # beat_offset = 0.5
        # aligned["beat_offset"]=(
        #     aligned["onset"]-aligned["beat_onset"]
        # )
        return aligned
    
    def process_singel_sample(self,melid):

        #database
        data_result = self.data.process_one(melid) #use info from class date
        metadata = data_result["metadata"]
        transcription = data_result["transcription"]
        melody_df = data_result["melody"]
        beats_df = data_result["beats"]
        #midi
        filename_solo = (transcription["filename_solo"])
        midi_result = self.midi.process_midi(filename_solo)
        #Alignment
        aligned_df = self.align(melody_df, beats_df)
        velocity_map = self.midi.extract_velocity(midi_result["path"])
        aligned_df["key"] = metadata.get("key", "Unknown")
        aligned_df["style"] = metadata.get("style", "Unknown")
        aligned_df["tempo"] = metadata.get("avgtempo", 120)
        # convert Nan into Null and df to dict
        melody_records = (melody_df.to_dict(orient="records")           )
        beat_records = (beats_df.to_dict(orient="records"))
        melody_records = (self.features.add_velocity(melody_records,velocity_map))
        melody_records = (self.features.add_articulation(melody_records))
        melody_records = (self.features.add_micro_timing(melody_records))
        melody_records = (self.features.add_chord_embedding(melody_records,beat_records))
        swing_ratio = (self.features.calculate_swing(melody_records))
        style_embedding = (self.features.style_embedding( metadata.get("style","OTHER")))
        metadata["swing_ratio"] = swing_ratio
        metadata["style_embedding"] = style_embedding




        aligned_records = (
            aligned_df.to_dict(orient="records")
            )
        
        # Unified sample
        sample = {
            "melid": melid,
            "metadata": metadata,
            "transcription":transcription,
            "midi": midi_result,
            "melody": melody_records,
            "beats": beat_records,
            "aligned": aligned_records}
        
        return sample
    
    def save_json(self, sample):
        # save processed singal sample as json
        melid = sample["melid"]
        output_path = (self.output_dir/f"solo_{melid:04d}.json")
        with output_path.open("w", encoding = "utf-8") as f:
            json.dump(sample,f,ensure_ascii=False,indent=2,allow_nan=False,default=str) 
            #ensure_ascii=False allow json save Chinese
        return output_path
    
    def process_all_samples(self):
        #process all samples in WJazzD.
        melids = self.data.get_all_melids()
        total = len(melids)
        print(f"Found {total} solos in WJazzD.")

        success_count = 0
        failed_count = 0
        failed_name = []
        for index, melid in enumerate(melids, start=1):
            print(f"[{index}/{total}]")
            print(f'Processing melid={melid}')
            try:
                sample = self.process_singel_sample(melid)
                ouput_path = self.save_json(sample)
                success_count += 1
                print(f"Performer: {sample['metadata']['performer']}")
                print(f"Title: {sample['metadata']['title']}")
                print(f"Melody: {len(sample['melody'])} events")
                print(f"Beats: {len(sample['beats'])} events")
                print(f"Aligned: {len(sample['aligned'])} events")
                print(f"Midi: {sample['midi']['filename']}")
                print(f"Saved: {ouput_path}")
            except Exception as e:
                failed_count += 1
                failed_name.append(melid)
                print(f"Error: {e}")
                traceback.print_exc()
                print("-" * 50)
        #Summary
        print("-" * 50)
        print("Processing Finishe")
        print("-" * 50)
        print(f"Successed: {success_count}")
        print(f'Failed: {failed_count}')
        if failed_name is not None:
            print(f"Failed melids: {failed_name}")


def main():
    print('-'*50)
    print("JassGen - WJazzD Preprocessing")
    print("-" * 50)
    print(f"Database: {DB_PATH}")
    print(f"Mid Dir: {MIDI_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    data=Data(DB_PATH)
    midi=Midi(MIDI_DIR)
    preprocess=Preprocess(data, midi,OUTPUT_DIR)
    preprocess.process_all_samples()
    print("Done!")

if __name__ == '__main__':
  main()

        




