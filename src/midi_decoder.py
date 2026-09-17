import json
from mido import MidiFile, MidiTrack, Message, MetaMessage



TICKS_PER_BEAT = 480



def token_value(token):

    return token.split("_")[-1]



def position_to_tick(position, tatum):

    """
    POSITION_x-y

    x = beat position
    y = subdivision

    """

    try:

        beat,sub = position.split("-")

        beat=int(beat)
        sub=int(sub)


        tick = (
            (beat-1)*480
            +
            (sub-1)*120
        )


        return tick


    except:

        return 0




def tokens_to_midi(tokens):


    mid=MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track=MidiTrack()

    mid.tracks.append(track)



    current_bar=0
    current_position=0

    velocity=90
    duration=120


    notes=[]



    for token in tokens:



        # ----------------
        # BAR
        # ----------------

        if token.startswith("<BAR"):


            try:

                current_bar=int(
                    token.split("_")[1]
                    .replace(">","")
                )


            except:

                pass



        # ----------------
        # POSITION
        # ----------------

        elif token.startswith(
            "POSITION"
        ):

            current_position=token_value(
                token
            )



        # ----------------
        # Velocity
        # ----------------

        elif token.startswith(
            "VELOCITY"
        ):

            velocity=int(
                float(
                    token_value(token)
                )
            )



        # ----------------
        # Duration
        # ----------------

        elif token.startswith(
            "DURATION"
        ):

            d=token_value(token)


            mapping={

                "32":60,
                "16":120,
                "8":240,
                "4":480,
                "2":960,
                "LONG":1920

            }


            duration=mapping.get(
                d,
                120
            )



        # ----------------
        # Pitch
        # ----------------

        elif token.startswith(
            "PITCH"
        ):


            pitch=int(
                float(
                    token_value(token)
                )
            )


            tick = (
                current_bar*1920
                +
                position_to_tick(
                    current_position,
                    None
                )
            )



            notes.append(
                {
                    "pitch":pitch,
                    "start":tick,
                    "duration":duration,
                    "velocity":velocity
                }
            )



    # ==========================
    # remove overlap
    # ==========================

    notes=sorted(
        notes,
        key=lambda x:x["start"]
    )


    # ==========================
    # Write midi
    # ==========================

    last_tick=0


    for n in notes:


        delta=n["start"]-last_tick


        track.append(
            Message(
                "note_on",
                note=n["pitch"],
                velocity=n["velocity"],
                time=max(delta,0)
            )
        )


        track.append(
            Message(
                "note_off",
                note=n["pitch"],
                velocity=0,
                time=n["duration"]
            )
        )


        last_tick=n["start"]



    return mid