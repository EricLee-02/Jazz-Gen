

from __future__ import annotations

import time
from pathlib import Path

import mido


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIDI_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WJazzD"
    / "midi"
    / "ArtPepper_Anthropology_FINAL.mid"
)

PORT_KEYWORD = "JazzGen_MIDI"


def find_output_port() -> str:
    ports = mido.get_output_names()

    print("Available MIDI outputs:")
    for port in ports:
        print("  ", port)

    for port in ports:
        if PORT_KEYWORD in port:
            return port

    raise RuntimeError("JazzGen_MIDI not found")


def send_cc(outport, control: int, value: int = 127):
    """
    Send a MIDI Control Change.
    CC100 = Record
    CC101 = Play
    CC102 = Stop
    """

    message = mido.Message(
        "control_change",
        channel=0,
        control=control,
        value=value,
    )

    outport.send(message)

    print(f"Sent CC{control} = {value}")


def main():

    if not MIDI_PATH.exists():
        raise FileNotFoundError(
            f"MIDI file not found:\n{MIDI_PATH}"
        )

    port_name = find_output_port()

    print("\n" + "=" * 60)
    print("JazzGen - Logic Play & Record Test")
    print("=" * 60)

    print(f"MIDI : {MIDI_PATH.name}")
    print(f"Port : {port_name}")
    print()

    midi = mido.MidiFile(MIDI_PATH)

    with mido.open_output(port_name) as outport:

        # ----------------------------------------------------
        # 1. Logic Record
        # ----------------------------------------------------

        print("1. Logic Record...")
        send_cc(outport, 100)

        time.sleep(1.0)

        # ----------------------------------------------------
        # 2. Logic Play
        # ----------------------------------------------------

        # print("2. Logic Play...")
        # send_cc(outport, 101)

        # time.sleep(0.5)

        # ----------------------------------------------------
        # 3. Send MIDI music
        # ----------------------------------------------------

        print("3. Sending MIDI...\n")

        message_count = 0

        for message in midi.play():

            if message.is_meta:
                continue

            outport.send(message)

            message_count += 1

        print(
            f"\nSent {message_count} MIDI messages."
        )

        # ----------------------------------------------------
        # 4. Stop Logic
        # ----------------------------------------------------

        time.sleep(0.5)

        print("4. Logic Stop...")
        send_cc(outport, 102)

    print("\nFinished.")


if __name__ == "__main__":
    main()