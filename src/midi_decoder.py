import json
from mido import MidiFile, MidiTrack, Message, MetaMessage


BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"

TOKEN_FILE = BASE_DIR + "/Output/generated_tokens.json"

OUTPUT_FILE = BASE_DIR + "/Output/generated_jazz.mid"


TICKS_PER_BEAT = 480



# =========================
# token value
# =========================

def get_value(token):

    value = token.split("_")[-1]

    try:
        return int(value)

    except:
        return value



# =========================
# duration
# =========================

def duration_to_ticks(token):


    duration_map = {

        "DURATION_32":0.125,

        "DURATION_16":0.25,

        "DURATION_8":0.5,

        "DURATION_4":1.0,

        "DURATION_2":2.0,

        "DURATION_1":4.0,

        "DURATION_LONG":4.0

    }


    beats = duration_map.get(
        token,
        0.25
    )


    return int(
        beats*TICKS_PER_BEAT
    )



# =========================
# calculate note position
# =========================

def calculate_tick(
        bar,
        beat,
        position
):

    """
    WJazzD:

    bar: 0-based

    beat: 1-4

    position:
        beat-tatum

    """

    tick = (

        bar
        *
        4
        *
        TICKS_PER_BEAT

    )


    # beat convert
    tick += (

        beat-1

    ) * TICKS_PER_BEAT



    # position

    if isinstance(position,str):

        try:

            beat_part, tatum_part = position.split("-")


            tatum=int(tatum_part)


            # 每拍4 subdivision

            tick += int(
                (tatum-1)
                *
                TICKS_PER_BEAT/4
            )


        except:

            pass



    return int(tick)





# =========================
# decode
# =========================

def tokens_to_midi(tokens):


    mid=MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track=MidiTrack()

    mid.tracks.append(track)



    # state

    bar=0

    beat=1

    position="1-1"


    velocity=80

    duration="DURATION_16"



    notes=[]



    for token in tokens:



        # -------------------
        # bar
        # -------------------

        if token.startswith("<BAR"):


            bar=get_value(token)



        elif token.startswith("BEAT"):


            beat=get_value(token)



        elif token.startswith("POSITION"):


            position=token.split("_")[-1]



        elif token.startswith("VELOCITY"):


            velocity=get_value(token)



        elif token.startswith("DURATION"):


            duration=token



        elif token.startswith("PITCH"):


            pitch=get_value(token)



            onset=calculate_tick(

                bar,
                beat,
                position

            )


            notes.append(

                {

                "pitch":pitch,

                "velocity":velocity,

                "start":onset,

                "end":
                onset+
                duration_to_ticks(duration)

                }

            )




    # =========================
    # write midi
    # =========================


    events=[]


    for note in notes:


        events.append(

            (
                note["start"],

                Message(

                    "note_on",

                    note=note["pitch"],

                    velocity=note["velocity"],

                    time=0

                )

            )

        )


        events.append(

            (
                note["end"],

                Message(

                    "note_off",

                    note=note["pitch"],

                    velocity=0,

                    time=0

                )

            )

        )



    # sort by absolute time

    events.sort(
        key=lambda x:x[0]
    )



    current_time=0


    for tick,msg in events:


        msg.time=int(
            tick-current_time
        )

        track.append(msg)

        current_time=tick



    return mid





# =========================
# main
# =========================

if __name__=="__main__":


    with open(
        TOKEN_FILE,
        "r"
    ) as f:

        tokens=json.load(f)



    print(
        "Tokens:",
        len(tokens)
    )


    midi=tokens_to_midi(tokens)


    midi.save(
        OUTPUT_FILE
    )


    print(
        "Saved:",
        OUTPUT_FILE
    )