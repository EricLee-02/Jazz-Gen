import json
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"

TOKEN_FILE = BASE_DIR + "/Output/generated_tokens.json"

OUTPUT_FILE = BASE_DIR + "/Output/generated_jazz.mid"

TICKS_PER_BEAT = 480



# ============================
# token value parser
# ============================

def get_value(token):

    value = token.split("_")[-1]

    try:
        return float(value)

    except:
        return value



# ============================
# duration token decoder
# ============================

def duration_to_beats(token):

    duration_map = {

        "DURATION_32":0.125,

        "DURATION_16":0.25,

        "DURATION_8":0.5,

        "DURATION_4":1.0,

        "DURATION_2":2.0,

        "DURATION_LONG":4.0

    }


    return duration_map.get(
        token,
        0.25
    )




# ============================
# token -> midi
# ============================

def tokens_to_midi(tokens):


    mid = MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track = MidiTrack()

    mid.tracks.append(track)



    velocity = 80

    duration = 0.25

    tempo = 120


    pending_notes=[]



    for token in tokens:


        # =====================
        # Tempo
        # =====================

        if token.startswith("TEMPO"):


            tempo=get_value(token)


            track.append(
                MetaMessage(
                    "set_tempo",
                    tempo=int(60000000/tempo),
                    time=0
                )
            )



        # =====================
        # Velocity
        # =====================

        elif token.startswith("VELOCITY"):


            velocity=int(
                get_value(token)
            )



        # =====================
        # Duration
        # =====================

        elif token.startswith("DURATION"):


            duration=duration_to_beats(
                token
            )



        # =====================
        # Pitch
        # =====================

        elif token.startswith("PITCH"):


            pitch=int(
                get_value(token)
            )


            pending_notes.append(
                {
                    "pitch":pitch,
                    "velocity":velocity,
                    "duration":duration
                }
            )



        # =====================
        # New bar
        # =====================

        elif token.startswith("<BAR"):


            write_notes(
                track,
                pending_notes
            )

            pending_notes=[]



    # =====================
    # remaining notes
    # =====================

    write_notes(
        track,
        pending_notes
    )


    return mid




# ============================
# write notes
# ============================

def write_notes(track,notes):


    for note in notes:


        ticks=int(
            note["duration"]
            *
            TICKS_PER_BEAT
        )


        track.append(
            Message(
                "note_on",
                note=note["pitch"],
                velocity=note["velocity"],
                time=0
            )
        )


        track.append(
            Message(
                "note_off",
                note=note["pitch"],
                velocity=0,
                time=ticks
            )
        )




# ============================
# main
# ============================

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