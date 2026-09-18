from mido import (
    MidiFile,
    MidiTrack,
    Message,
    MetaMessage
)



TICKS_PER_BEAT = 480



# ==============================
# token value
# ==============================

def token_value(token):

    return token.split("_",1)[-1]




# ==============================
# Duration
# ==============================

def duration_to_tick(value):

    mapping={

        "32":60,
        "16":120,
        "8":240,
        "4":480,
        "2":960,
        "1":1920,
        "LONG":1920

    }

    return mapping.get(
        value,
        120
    )




# ==============================
# Swing position
# ==============================

def position_to_tick(position):


    try:
        beat = int(beat)
        tatum = int(tatum)
        tick = (beat-1)*480
        if tatum == 1:
            sub = 0
        elif tatum == 2:
            sub = 240
        elif tatum == 3:
            sub = 320
        else:
            sub = 360
        tick +=sub
        return tick


    except:

        return 0





# ==============================
# Micro timing
# ==============================

def apply_micro(tick,micro):
    if micro=="MICRO_LATE":
        tick += 20
    elif micro=="MICRO_EARLY":
        tick -= 10
    return max(tick,0)






# ==============================
# Articulation
# ==============================

def apply_articulation(duration,artic):


    if artic=="ARTIC_staccato":

        return int(duration*0.85)


    elif artic=="ARTIC_legato":

        return int(duration*1.1)


    else:

        return duration







# ==============================
# Main decoder
# ==============================


def tokens_to_midi(tokens):


    mid=MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track=MidiTrack()

    mid.tracks.append(track)



    # tempo

    track.append(
        MetaMessage(
            "set_tempo",
            tempo=500000,
            time=0
        )
    )



    bar=0

    position="1-1"

    velocity=90

    duration=120

    micro=None

    articulation=None
    current_beat = 1
    current_tatum = 1


    events=[]



    for token in tokens:



        # ----------------
        # BAR
        # ----------------

        if token.startswith("BAR"):

            try:

                bar=int(
                    token.split("_")[1]
                )

            except:

                pass

        elif token.startswith("BEAT"):
            current_beat = int(token_value(token))
        
        elif token.startswith("TATUM"):
            current_tatum = int(token_value(token))



        # ----------------
        # Position
        # ----------------

        elif token.startswith(
            "POSITION"
        ):

            position=token_value(token)



        # ----------------
        # velocity
        # ----------------

        elif token.startswith(
            "VELOCITY"
        ):

            velocity=int(
                token_value(token)
            )



        # ----------------
        # duration
        # ----------------

        elif token.startswith(
            "DURATION"
        ):


            duration=duration_to_tick(
                token_value(token)
            )



        # ----------------
        # micro timing
        # ----------------

        elif token.startswith(
            "MICRO"
        ):

            micro=token



        # ----------------
        # articulation
        # ----------------

        elif token.startswith(
            "ARTIC"
        ):

            articulation=token



        # ----------------
        # pitch
        # ----------------

        elif token.startswith(
            "PITCH"
        ):


            pitch=int(
                token_value(token)
            )



            start=(bar*4*TICKS_PER_BEAT+position_to_tick(current_beat,current_tatum,position))


            start=apply_micro(
                start,
                micro
            )



            dur=apply_articulation(
                duration,
                articulation
            )



            events.append(

                {
                    "type":"on",
                    "tick":start,
                    "pitch":pitch,
                    "velocity":velocity
                }

            )


            events.append(

                {
                    "type":"off",
                    "tick":start+dur,
                    "pitch":pitch,
                    "velocity":0
                }

            )



            micro=None




    # ==============================
    # Sort events
    # ==============================

    events.sort(
        key=lambda x:
        (
            x["tick"],
            0 if x["type"]=="off" else 1
        )
    )




    # ==============================
    # Write midi
    # ==============================


    last_tick=0



    for e in events:


        delta=e["tick"]-last_tick


        track.append(

            Message(

                "note_on"
                if e["type"]=="on"
                else
                "note_off",

                note=e["pitch"],

                velocity=e["velocity"],

                time=max(
                    delta,
                    0
                )

            )

        )


        last_tick=e["tick"]




    return mid