import json
from mido import MidiFile, MidiTrack, Message, MetaMessage


BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"

TOKEN_FILE = BASE_DIR + "/Output/generated_tokens.json"

OUTPUT_FILE = BASE_DIR + "/Output/generated_jazz.mid"


TICKS_PER_BEAT = 480



# ==========================
# token parser
# ==========================

def get_value(token):

    value = token.split("_")[-1]

    try:
        return int(value)

    except:

        return value



# ==========================
# duration
# ==========================

def duration_to_beats(token):

    table={

        "DURATION_32":0.125,

        "DURATION_16":0.25,

        "DURATION_8":0.5,

        "DURATION_4":1.0,

        "DURATION_2":2.0,

        "DURATION_1":4.0,

        "DURATION_LONG":4.0
    }


    return table.get(
        token,
        0.25
    )



# ==========================
# micro timing
# ==========================

def micro_offset(token):

    if token=="MICRO_LATE":

        return 30


    elif token=="MICRO_EARLY":

        return -30


    return 0



# ==========================
# swing deformation
# ==========================

def apply_swing(tick, swing_ratio):

    """
    simple jazz swing

    ratio >1 means swing feel
    """

    if swing_ratio <=1.2:

        return tick


    beat_position = tick % 480


    # second eighth note delay

    if 240 <= beat_position < 480:

        shift = int(
            (swing_ratio-1)
            *
            40
        )

        tick += shift


    return tick



# ==========================
# tokens -> midi
# ==========================


def tokens_to_midi(tokens):


    mid=MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track=MidiTrack()

    mid.tracks.append(track)



    # state

    bar=0

    beat=0

    tatum=0

    position=None


    velocity=80

    duration="DURATION_16"

    micro="MICRO_GRID"

    swing_ratio=1.0


    pending_notes=[]



    # current time

    current_tick=0



    for token in tokens:



        # =====================
        # metadata
        # =====================

        if token.startswith("SWING_RATIO"):

            try:

                swing_ratio=float(
                    token.split("_")[-1]
                )

            except:

                pass



        # =====================
        # position
        # =====================

        elif token.startswith("<BAR"):


            bar=get_value(token)



        elif token.startswith("BEAT"):

            beat=get_value(token)



        elif token.startswith("TATUM"):

            tatum=get_value(token)



        elif token.startswith("POSITION"):

            position=get_value(token)



        # =====================
        # note feature
        # =====================


        elif token.startswith("VELOCITY"):


            velocity=int(
                get_value(token)
            )



        elif token.startswith("DURATION"):

            duration=token



        elif token.startswith("MICRO"):

            micro=token



        elif token.startswith("PITCH"):


            pitch=int(
                get_value(token)
            )


            # =====================
            # calculate onset
            # =====================


            onset = (

                bar
                *
                4
                *
                TICKS_PER_BEAT

                +

                beat
                *
                TICKS_PER_BEAT

                +

                tatum
                *
                (
                    TICKS_PER_BEAT/4
                )

            )


            onset += micro_offset(
                micro
            )


            onset=int(
                apply_swing(
                    onset,
                    swing_ratio
                )
            )



            pending_notes.append(

                {

                "pitch":pitch,

                "velocity":velocity,

                "duration":duration,

                "onset":onset

                }

            )



    # =====================
    # write midi
    # =====================


    events=[]


    for note in pending_notes:


        start=note["onset"]

        length=int(

            duration_to_beats(
                note["duration"]
            )

            *

            TICKS_PER_BEAT

        )


        end=start+length


        events.append(

            (
                start,
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
                end,

                Message(
                    "note_off",
                    note=note["pitch"],
                    velocity=0,
                    time=0
                )

            )

        )



    # sort by time

    events.sort(
        key=lambda x:x[0]
    )


    last_time=0


    for time,msg in events:


        msg.time=int(
            time-last_time
        )

        track.append(msg)

        last_time=time



    return mid




# ==========================
# main
# ==========================


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