import json
import os
import mido
from mido import Message, MidiFile, MidiTrack


# =========================
# Config
# =========================

BASE_DIR = "/content/drive/MyDrive/JazzGen_Data"

TOKEN_FILE = (
    BASE_DIR +
    "/generated/generated_tokens.json"
)


OUTPUT_FILE = (
    BASE_DIR +
    "/generated/generated_jazz.mid"
)


TICKS_PER_BEAT = 480



# =========================
# Token Parser
# =========================


def parse_token(token):

    """
    将token解析成事件

    返回:
    ("type",value)

    """

    parts = token.split("_")


    if len(parts) < 2:
        return None,None


    key = parts[0].lower()


    value = parts[-1]


    try:
        value = int(value)
    except:
        pass



    # Note

    if "note" in key:

        return "note", value



    # pitch

    if "pitch" in key:

        return "note", value



    # velocity

    if "velocity" in key:

        return "velocity", value



    # duration

    if "duration" in key:

        return "duration", value



    # time shift

    if "time" in key:

        return "time", value


    return None,None




# =========================
# Decode
# =========================


def tokens_to_midi(tokens):


    mid = MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track = MidiTrack()

    mid.tracks.append(track)



    current_velocity = 80

    current_duration = 480


    current_time = 0



    active_notes=[]



    for token in tokens:


        event,value = parse_token(token)


        if event is None:
            continue



        # velocity

        if event=="velocity":

            current_velocity = max(
                1,
                min(value,127)
            )



        # duration

        elif event=="duration":

            current_duration=value



        # note

        elif event=="note":


            pitch=value


            if not isinstance(
                pitch,
                int
            ):
                continue


            pitch=max(
                0,
                min(
                    pitch,
                    127
                )
            )


            # note on

            track.append(
                Message(
                    "note_on",
                    note=pitch,
                    velocity=current_velocity,
                    time=0
                )
            )


            # note off

            track.append(
                Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    time=current_duration
                )
            )



        # time shift

        elif event=="time":

            current_time += value



    return mid




# =========================
# Main
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



    midi=tokens_to_midi(
        tokens
    )


    midi.save(
        OUTPUT_FILE
    )


    print(
        "Saved:",
        OUTPUT_FILE
    )