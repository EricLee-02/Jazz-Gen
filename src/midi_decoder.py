"""
Decode JazzGen melody tokens into MIDI.

Token format:

<BOS>
KEY_xxx
AVGTEMPO_xxx

BAR_x
PERIOD_x
BEAT_x
DIVISION_x
TATUM_x

PITCH_x
DURATION_x
VELOCITY_x
ARTIC_x
MICRO_x

"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


# =========================
# MIDI constants
# =========================

TICKS_PER_BEAT = 480


# =========================
# Duration mapping
# =========================

def duration_to_ticks(duration):

    """
    Convert token duration to MIDI ticks

    DURATION:
    
    1  = whole note
    2  = half
    4  = quarter
    8  = eighth
    16 = sixteenth

    """

    mapping = {

        "1": TICKS_PER_BEAT * 4,

        "2": TICKS_PER_BEAT * 2,

        "4": TICKS_PER_BEAT,

        "8": TICKS_PER_BEAT // 2,

        "16": TICKS_PER_BEAT // 4,

        "32": TICKS_PER_BEAT // 8,

    }


    if duration not in mapping:

        raise ValueError(
            f"Unknown duration token: {duration}"
        )


    return mapping[duration]



# =========================
# Note object
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

        self.bar = bar

        self.beat = beat

        self.division = division

        self.tatum = tatum

        self.pitch = pitch

        self.duration = duration

        self.velocity = velocity

        self.articulation = articulation

        self.micro = micro



# =========================
# Token parser
# =========================


def parse_notes(tokens):


    notes=[]


    current={

        "bar":None,

        "beat":None,

        "division":None,

        "tatum":None,

    }


    pending=None



    for idx,token in enumerate(tokens):


        if not isinstance(token,str):

            continue



        if token in [

            "<BOS>",
            "<EOS>",
            "<PAD>"

        ]:

            continue



        field,value = (

            token.split("_",1)

            if "_" in token

            else

            (token,None)

        )



        # -----------------
        # Time information
        # -----------------

        if field=="BAR":

            current["bar"]=int(value)


        elif field=="BEAT":

            current["beat"]=int(value)


        elif field=="DIVISION":

            current["division"]=int(value)



        elif field=="TATUM":

            current["tatum"]=int(value)



        # -----------------
        # New note
        # -----------------

        elif field=="PITCH":


            pending={

                "bar":current["bar"],

                "beat":current["beat"],

                "division":current["division"],

                "tatum":current["tatum"],


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



            # note event complete

            if pending["duration"]:

                notes.append(

                    Note(**pending)

                )

                pending=None



    return notes




# =========================
# Position convert
# =========================


def note_to_tick(note):


    """
    Convert BAR/BEAT/DIVISION/TATUM
    to absolute MIDI tick
    """


    ticks_per_division = TICKS_PER_BEAT // 4


    tick = (

        note.bar
        *
        4
        *
        TICKS_PER_BEAT

    )


    tick += (

        (note.beat-1)
        *
        TICKS_PER_BEAT

    )


    tick += (

        (note.division-1)
        *
        ticks_per_division

    )


    tick += (

        (note.tatum-1)
        *
        ticks_per_division

    )


    return tick




# =========================
# MIDI export
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


        end=start+duration_to_ticks(
            note.duration
        )


        events.append(

            (

                start,

                "on",

                note

            )

        )


        events.append(

            (

                end,

                "off",

                note

            )

        )



    events.sort(

        key=lambda x:x[0]

    )


    last_tick=0



    for tick,event,note in events:


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