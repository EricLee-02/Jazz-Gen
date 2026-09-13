import json
from collections import Counter


input_file = (
    "/Volumes/My Passport/Jazz Gen/data/processed/"
    "ireal_harmony/ireal_harmony_form.json"
)



with open(
    input_file,
    "r",
    encoding="utf8"
) as f:

    songs=json.load(f)



unknown_counter=Counter()

total_events=0
unknown_events=0



for song in songs:


    events=song.get(
        "events",
        []
    )


    for e in events:


        total_events+=1


        raw=e.get(
            "raw_symbol",
            None
        )


        chord=e.get(
            "raw_symbol",
            None
        )


        if chord in [
            None,
            "",
            "<UNK>"
        ]:

            unknown_events+=1


            unknown_counter[
                str(raw)
            ] +=1





print("======================")
print("Total events:",total_events)

print(
    "Unknown events:",
    unknown_events
)


ratio=unknown_events/total_events*100


print(
    "Unknown ratio:",
    round(ratio,4),
    "%"
)


print("======================")
print("Unknown chord examples:")



for k,v in unknown_counter.most_common(50):

    print(
        k,
        ":",
        v
    )