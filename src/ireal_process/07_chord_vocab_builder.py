import json
from pathlib import Path

input_file = "/Volumes/My Passport/Jazz Gen/data/processed/ireal_harmony/ireal_harmony_form.json"
output_dir =Path("/Volumes/My Passport/Jazz Gen/data/processed/ireal_harmony/vocab")
output_dir.mkdir(exist_ok=True)



# ======================================================
# Vocabulary Builder
# ======================================================

class VocabularyBuilder:
    def __init__(self):
        self.special_tokens=["<PAD>", "<UNK>","<MASK>"]

    def build(self,items):

        vocab=self.special_tokens + sorted(list(set(items)))
        token_to_id={token:i for i,token in enumerate(vocab)}
        id_to_token={i:token for token,i in token_to_id.items()}


        return {
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "size": len(token_to_id)
        }
    
    def normalize_chord(chord):
        if chord is None:
            return None
        
        chord = chord.strip()
        if chord == "":
            return None
        
        if chord.startswith("N") or chord.startswith("x"):
            return None
        
        if chord in ["x","S","J","n","nn","r","N2"]:
            return None
        
        if "U" in chord:
            if chord == "U":
                return None
            chord = chord.replace("U","")

            if chord == "":
                return None
        
        chord = chord.replace("h","-7b5")

        return chord




# ======================================================
# Load JSON
# ======================================================

with open(input_file,"r",encoding="utf8") as f:
    songs=json.load(f)

chords=set()
durations=set()
beats=set()
sections=set()
times=set()



# ======================================================
# Extract
# ======================================================

for song in songs:
    # -----------------
    # time signature
    # -----------------

    ts=song.get("time_signature")


    if ts:
        times.add(f"{ts[0]}/{ts[1]}")



    # -----------------
    # sections
    # -----------------

    form=song.get("form",{})
    for sec in form.get("sections",[]):
        sections.add(sec["name"])

    # -----------------
    # events
    # -----------------

    for event in song.get( "events",[]):
        raw_symbol = event.get("raw_symbol")
        clean_symbol = VocabularyBuilder.normalize_chord(raw_symbol)
        if clean_symbol is not None:
            chords.add(clean_symbol)
        duration=event.get("duration")
        if duration is not None:
            durations.add(str(duration))
        beat=event.get("beat_start")
        if beat is not None:
            beats.add(str(beat))





builder=VocabularyBuilder()



vocabs={
    "chord_vocab":builder.build(chords),
    "duration_vocab":builder.build(durations),
    "beat_vocab":builder.build(beats),
    "section_vocab":builder.build(sections),
    "time_signature_vocab":builder.build(times)
}


# ======================================================
# Save
# ======================================================

for name,data in vocabs.items():
    path=output_dir / f"{name}.json"
    with open( path, "w", encoding="utf8") as f:
        json.dump(data,f,indent=4, ensure_ascii=False)


    print(name, "size:", data["size"])


print("Vocabulary finished")

total_events=0
total_bars=0


for song in songs:

    total_events += len(song.get("events",[]))

    total_bars += (song.get("form",{}).get("total_bars",0))


print("Total events:",total_events)
print("Total bars:",total_bars)