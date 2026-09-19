"""
JazzGen MIDI Decoder

Decode:

BAR
BEAT
DIVISION
TATUM

PITCH
DURATION
VELOCITY
ARTIC
MICRO

into MIDI

"""


import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage



# =========================
# MIDI settings
# =========================


TICKS_PER_BEAT = 480



# =========================
# Duration
# =========================


def duration_to_ticks(duration):


    mapping={

        "1": TICKS_PER_BEAT*4,

        "2": TICKS_PER_BEAT*2,

        "4": TICKS_PER_BEAT,

        "8": TICKS_PER_BEAT//2,

        "16": TICKS_PER_BEAT//4,

        "32": TICKS_PER_BEAT//8,

    }


    return mapping.get(
        duration,
        TICKS_PER_BEAT//4
    )



# =========================
# Note
# =========================


class Note:


    def __init__(
        self,
        bar,
        beat,
        division,
        tatum,
        pitch,
        duration,
        velocity,
        articulation=None,
        micro=None
    ):

        self.bar=bar
        self.beat=beat
        self.division=division
        self.tatum=tatum

        self.pitch=pitch
        self.duration=duration

        self.velocity=velocity

        self.articulation=articulation
        self.micro=micro




# =========================
# Parse tokens
# =========================


def parse_notes(tokens):


    notes=[]


    current={

        "bar":0,
        "beat":1,
        "division":1,
        "tatum":1

    }


    pending=None



    for token in tokens:


        if not isinstance(token,str):
            continue



        if token in [
            "<BOS>",
            "<EOS>",
            "<PAD>"
        ]:
            continue



        if "_" not in token:
            continue



        field,value=token.split("_",1)



        if field=="BAR":

            current["bar"]=int(value)


        elif field=="BEAT":

            current["beat"]=int(value)


        elif field=="DIVISION":

            current["division"]=int(value)


        elif field=="TATUM":

            current["tatum"]=int(value)



        elif field=="PITCH":


            pending={

                **current,

                "pitch":int(value),

                "duration":None,

                "velocity":100,

                "articulation":None,

                "micro":None

            }



        elif field=="DURATION":

            if pending:
                pending["duration"]=value



        elif field=="VELOCITY":

            if pending:
                pending["velocity"]=int(value)



        elif field=="ARTIC":

            if pending:
                pending["articulation"]=value



        elif field=="MICRO":


            if pending:

                pending["micro"]=value


                if pending["duration"]:


                    notes.append(
                        Note(**pending)
                    )

                    pending=None



    return notes




# =========================
# Time conversion
# =========================


def note_to_tick(note):


    """
    Convert hierarchical musical position

    BAR
     |
     BEAT
       |
       DIVISION
          |
          TATUM

    """


    # bar

    tick = (

        note.bar
        *
        4
        *
        TICKS_PER_BEAT

    )


    # beat

    tick += (

        note.beat-1
    ) * TICKS_PER_BEAT



    # division

    # division is quarter subdivision

    tick += (

        note.division-1
    ) * (

        TICKS_PER_BEAT//4

    )



    # tatum is smaller subdivision

    tick += (

        note.tatum-1
    ) * (

        TICKS_PER_BEAT//16

    )



    # micro timing


    if note.micro:


        if "EARLY" in note.micro:

            tick -= 15


        elif "LATE" in note.micro:

            tick +=15



    return max(
        int(tick),
        0
    )





# =========================
# articulation
# =========================


def apply_articulation(
    ticks,
    articulation
):


    if articulation=="staccato":

        return int(
            ticks*0.55
        )


    elif articulation=="tenuto":

        return int(
            ticks*1.1
        )


    return ticks




# =========================
# Export MIDI
# =========================


def tokens_to_midi(
    tokens,
    tempo_bpm=120
):


    notes=parse_notes(tokens)



    midi=MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )



    track=MidiTrack()

    midi.tracks.append(track)



    track.append(

        MetaMessage(
            "set_tempo",
            tempo=mido.bpm2tempo(
                tempo_bpm
            )
        )

    )



    events=[]



    for note in notes:


        start=note_to_tick(note)



        duration=duration_to_ticks(
            note.duration
        )



        duration=apply_articulation(
            duration,
            note.articulation
        )



        end=start+duration



        events.append(
            (
                start,
                1,
                "on",
                note
            )
        )



        events.append(
            (
                end,
                0,
                "off",
                note
            )
        )




    # off before on

    events.sort(
        key=lambda x:(x[0],x[1])
    )



    last_tick=0



    for tick,_,event,note in events:


        delta=tick-last_tick


        last_tick=tick



        if event=="on":


            track.append(

                Message(
                    "note_on",
                    note=note.pitch,
                    velocity=note.velocity,
                    time=delta
                )

            )


        else:


            track.append(

                Message(
                    "note_off",
                    note=note.pitch,
                    velocity=0,
                    time=delta
                )

            )



    return midi