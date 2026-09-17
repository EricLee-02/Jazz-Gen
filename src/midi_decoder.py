from mido import MidiFile, MidiTrack, Message, MetaMessage



TICKS_PER_BEAT = 480



# ==================================
# token value
# ============================== ====

def get_value(token):

    value = token.split("_")[-1]

    try:
        return int(value)

    except:

        try:
            return float(value)

        except:
            return value




# ==================================
# duration
# ==================================

def duration_to_ticks(token):


    duration_map = {


        # beat单位

        "DURATION_32":0.125,

        "DURATION_16":0.25,

        "DURATION_8":0.5,

        "DURATION_4":1.0,

        "DURATION_2":2.0,

        "DURATION_LONG":4.0


    }


    beat = duration_map.get(
        token,
        0.25
    )


    return int(
        beat*TICKS_PER_BEAT
    )





# ==================================
# position -> tick
# ==================================

def calculate_tick(
        bar,
        position
):


    """
    WJazzD position:

    1-1
    1-2
    2-1
    3-4


    first:
        beat

    second:
        tatum subdivision

    """


    tick = (

        bar
        *
        4
        *
        TICKS_PER_BEAT

    )



    if position is None:

        return tick



    try:


        beat,tatum = position.split("-")


        beat=int(beat)

        tatum=int(tatum)



        # beat position

        tick += (

            beat-1

        ) * TICKS_PER_BEAT



        # subdivision

        tick += (

            tatum-1

        ) * (

            TICKS_PER_BEAT/4

        )



    except Exception as e:


        print(
            "position error:",
            position,
            e
        )



    return int(tick)





# ==================================
# swing
# ==================================

def apply_swing(
        tick,
        swing_ratio
):


    """
    only move off beat eighth notes

    """


    if swing_ratio <= 1.1:

        return tick



    beat_pos = tick % TICKS_PER_BEAT



    # second eighth note

    if 200 < beat_pos < 400:


        delay = int(

            (swing_ratio-1)

            *

            80

        )


        tick += delay



    return tick






# ==================================
# micro timing
# ==================================

def apply_micro(
        tick,
        micro
):


    if micro=="MICRO_LATE":

        tick += 15


    elif micro=="MICRO_EARLY":

        tick -= 15


    return tick






# ==================================
# main decoder
# ==================================

def tokens_to_midi(tokens):


    midi=MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track=MidiTrack()

    midi.tracks.append(track)



    # current state


    bar=0

    position="1-1"

    velocity=80

    duration="DURATION_16"

    micro="MICRO_GRID"

    swing_ratio=1.0



    notes=[]



    for token in tokens:



        # --------------------------
        # metadata
        # --------------------------

        if token.startswith(
            "SWING_RATIO"
        ):


            swing_ratio=float(
                token.split("_")[-1]
            )




        # --------------------------
        # position
        # --------------------------

        elif token.startswith(
            "<BAR"
        ):


            bar=get_value(token)



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




        # --------------------------
        # note
        # --------------------------

        elif token.startswith(
            "PITCH"
        ):


            pitch=get_value(token)



            start=calculate_tick(

                bar,

                position

            )


            # swing

            start=apply_swing(

                start,

                swing_ratio

            )


            # micro

            start=apply_micro(

                start,

                micro

            )



            end=(

                start

                +

                duration_to_ticks(
                    duration
                )

            )



            notes.append(

                {

                "pitch":pitch,

                "velocity":velocity,

                "start":start,

                "end":end

                }

            )




    print(
        "Generated notes:",
        len(notes)
    )



    print(
        "First notes:",
        notes[:10]
    )




    # ==================================
    # events
    # ==================================


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




    # sort by time

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