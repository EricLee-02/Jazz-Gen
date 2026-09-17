import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage



TICKS_PER_BEAT = 480



# ==============================
# token parser
# ==============================

def get_value(token):

    value = token.split("_")[-1]

    try:
        return int(value)

    except:
        try:
            return float(value)
        except:
            return value



# ==============================
# duration
# ==============================

def duration_to_tick(token):

    duration_map = {

        "DURATION_32":0.125,

        "DURATION_16":0.25,

        "DURATION_8":0.5,

        "DURATION_4":1.0,

        "DURATION_2":2.0,

        "DURATION_1":4.0,

        "DURATION_LONG":4.0

    }


    beat = duration_map.get(
        token,
        0.25
    )


    return int(
        beat*TICKS_PER_BEAT
    )



# ==============================
# swing
# ==============================

def apply_swing(tick, swing_ratio):

    """
    Jazz swing:
    
    straight:
        1:1

    swing:
        2:1
        3:1


    only affect off-beat eighth notes

    """


    if swing_ratio <= 1.15:

        return tick



    beat_position = tick % TICKS_PER_BEAT



    # eighth note second half

    if 200 <= beat_position <= 350:


        delay = int(

            (swing_ratio-1)

            *

            80

        )


        tick += delay



    return tick




# ==============================
# micro timing
# ==============================

def apply_micro(tick, micro):


    if micro=="MICRO_LATE":

        tick += 20


    elif micro=="MICRO_EARLY":

        tick -= 20


    return tick




# ==============================
# calculate position
# ==============================

def calculate_tick(
        bar,
        beat,
        position
):


    """

    bar:
        0 based


    beat:
        1-4


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


    tick += (

        beat-1

    ) * TICKS_PER_BEAT




    if position:


        try:

            beat_id,sub = position.split("-")

            sub=int(sub)



            tick += int(

                (sub-1)

                *

                TICKS_PER_BEAT/4

            )


        except:

            pass



    return tick




# ==============================
# tokens -> MIDI
# ==============================


def tokens_to_midi(tokens):


    midi = MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track=MidiTrack()

    midi.tracks.append(track)



    # state

    bar=0

    beat=1

    position="1-1"


    velocity=80

    duration="DURATION_16"

    micro="MICRO_GRID"

    swing_ratio=1.0


    notes=[]



    for token in tokens:



        # --------------------
        # metadata
        # --------------------

        if token.startswith(
            "SWING_RATIO"
        ):


            try:

                swing_ratio=float(
                    token.split("_")[-1]
                )

            except:

                pass




        elif token.startswith(
            "<BAR"
        ):


            bar=get_value(token)




        elif token.startswith(
            "BEAT"
        ):


            beat=get_value(token)




        elif token.startswith(
            "POSITION"
        ):


            position=token.split("_")[1]




        elif token.startswith(
            "VELOCITY"
        ):


            velocity=get_value(token)




        elif token.startswith(
            "DURATION"
        ):


            duration=token




        elif token.startswith(
            "MICRO"
        ):


            micro=token




        elif token.startswith(
            "PITCH"
        ):


            pitch=get_value(token)



            onset=calculate_tick(

                bar,

                beat,

                position

            )


            # swing

            onset=apply_swing(

                onset,

                swing_ratio

            )


            # micro timing

            onset=apply_micro(

                onset,

                micro

            )



            notes.append(

                {

                "pitch":pitch,

                "velocity":velocity,

                "start":onset,

                "end":
                onset+
                duration_to_tick(
                    duration
                )

                }

            )



    # =====================
    # write MIDI events
    # =====================


    events=[]


    for n in notes:


        events.append(

            (

            n["start"],

            Message(
                "note_on",
                note=n["pitch"],
                velocity=n["velocity"],
                time=0
            )

            )

        )



        events.append(

            (

            n["end"],

            Message(
                "note_off",
                note=n["pitch"],
                velocity=0,
                time=0
            )

            )

        )




    # chronological order

    events.sort(
        key=lambda x:x[0]
    )



    current=0


    for tick,msg in events:


        msg.time=max(
            0,
            tick-current
        )


        track.append(msg)


        current=tick



    return midi