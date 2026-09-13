import json
import re

from pyRealParser import Tune



raw_file = "/Volumes/My Passport/Jazz Gen/data/processed/ireal_json/jazz1400.json"

embedding_file = "/Volumes/My Passport/Jazz Gen/data/processed/ireal_harmony/ireal_harmony_embedding.json"


output_file = "/Volumes/My Passport/Jazz Gen/data/processed/ireal_harmony/ireal_harmony_form.json"



# ======================================================
# Form Builder
# ======================================================

class FormBuilder:
    # --------------------------------------------------
    # 判断12小节Blues
    # --------------------------------------------------
    def detect_blues(self,tune):
        total_bars=len(tune.measures_as_strings)
        if total_bars == 12:
            return True
        return False


    # --------------------------------------------------
    # Build
    # --------------------------------------------------
    def build(self,tune):
        chord_string=tune.chord_string
        total_bars=len(tune.measures_as_strings)
        sections=[]
        # ==================================================
        # Section markers
        # ==================================================
        markers=[]
        for m in re.finditer(r"\*([A-Z])",chord_string):
            markers.append({
                "name":m.group(1),
                "pos":m.start()
            })

        # ==================================================
        # 没有section标记
        # ==================================================
        if len(markers)==0:
            if self.detect_blues(tune):
                sections.append({
                    "name":"Blues_12bars",
                    "start_bar":1,
                    "end_bar":total_bars
                })
            else:
                sections.append({
                    "name":"A",
                    "start_bar":1,
                    "end_bar":total_bars
                })

            return {
                "total_bars":total_bars,
                "time_signature":tune.time_signature,
                "sections":sections
            }




        # ==================================================
        # 有section marker
        # ==================================================
        counter={}
        current_bar=1
        for i,marker in enumerate(markers):
            name=marker["name"]
            counter.setdefault(name,0)
            start_pos=marker["pos"]
            if i+1 < len(markers):
                end_pos=markers[i+1]["pos"]
            else:
                end_pos=len(chord_string)
            raw_segment=chord_string[start_pos:end_pos]



            # ==================================================
            # Repeat Section
            # ==================================================

            if ("{" in raw_segment and "}" in raw_segment):
                repeat_match=re.search(r"\{(.*?)\}",raw_segment,re.S)

                if repeat_match:
                    repeat_body=(repeat_match.group(1))
                    repeat_bars=(repeat_body.count("|")+1)

                    # 第一次
                    counter[name]+=1
                    sections.append({
                        "name":f"{name}_{counter[name]}",
                        "start_bar":current_bar,
                        "end_bar":current_bar +repeat_bars - 1
                    })
                    current_bar+=repeat_bars
                    # 第二次
                    counter[name]+=1
                    sections.append({
                        "name":f"{name}_{counter[name]}",
                        "start_bar":current_bar,
                        "end_bar":current_bar+repeat_bars-1
                    })
                    current_bar+=repeat_bars
                    continue

            # ==================================================
            # 普通Section
            # ==================================================

            bar_count=(raw_segment.count("|") +1)
            counter[name]+=1
            start_bar=current_bar
            end_bar=(start_bar+ bar_count -1)

            if i==len(markers)-1:
                end_bar=total_bars
            end_bar=min( end_bar,total_bars)
            sections.append({
                "name":f"{name}_{counter[name]}",
                "start_bar":start_bar,
                "end_bar":end_bar
            })
            current_bar=end_bar+1


        return {
            "total_bars":total_bars,
            "time_signature":tune.time_signature,
            "sections":sections
        }




# ======================================================
# Event Section Mapping
# ======================================================

def assign_section(events,form):
    sections=form["sections"]
    for event in events:

        bar=event.get( "bar" )
        event["section_from"]="Unknown"

        if bar is None:
            continue

        for sec in sections:
            if (sec["start_bar"]<= bar<=sec["end_bar"]):
                event["section_from"]=sec["name"]
                break





# ======================================================
# Load
# ======================================================

with open(raw_file,"r",encoding="utf8") as f:
    raw_songs=json.load(f)


with open(embedding_file,"r",encoding="utf8") as f:
    songs=json.load(f)




# ======================================================
# URL index
# ======================================================

song_url_map={}

for item in raw_songs:

    key=(
        item["song_info"]["title"].lower(),
        item["song_info"]["composer"].lower()

    )
    song_url_map[key]=item["ireal_url"]

# ======================================================
# Process
# ======================================================

builder=FormBuilder()
processed=0
errors=[]
total=len(songs)

for song in songs:

    key=(

        song["title"].lower(),

        song["composer"].lower()

    )
    if key not in song_url_map:
        errors.append({
            "title":song["title"],
            "error":"url not found"
        })

        continue




    try:
        tune_list=Tune.parse_ireal_url(song_url_map[key])

        tune=tune_list[0]

        form=builder.build( tune)

        song["time_signature"]=(form["time_signature"])

        song["form"]={
            "total_bars":form["total_bars"],
            "sections": form["sections"]
        }

        assign_section(song.get("events",[]), form)

        processed+=1
        print(
            f"Processed: {song['title']} "
            f"({processed}/{total})"
        )

    except Exception as e:

        errors.append({
            "title":song["title"],
            "error":str(e)
        })

        print("ERROR:",song["title"],e)


# ======================================================
# Save
# ======================================================

with open(output_file,"w",encoding="utf8") as f:
    json.dump(songs, f,indent=4,ensure_ascii=False)




if errors:
    error_file="/Volumes/My Passport/Jazz Gen/data/processed/ireal_harmony/ireal_form_errors.json"

    with open(error_file, "w",encoding="utf8") as f:
        json.dump(errors,f,indent=4,ensure_ascii=False)


print("====================")
print("Finished:",processed)
print("Errors:",len(errors))