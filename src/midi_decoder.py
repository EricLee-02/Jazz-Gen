import json
from mido import MidiFile, MidiTrack, Message


JSON_FILE = "/path/to/song.json"

OUTPUT_MIDI = "/path/to/output.mid"


TICKS_PER_BEAT = 480



def json_to_midi(json_file, output_file):


    with open(json_file,"r") as f:
        data=json.load(f)


    midi=MidiFile(
        ticks_per_beat=TICKS_PER_BEAT
    )


    track=MidiTrack()

    midi.tracks.append(track)



    melody=data["melody"]


    events=[]


    for note in melody:


        pitch=int(note["pitch"])


        velocity=int(
            note.get(
                "velocity",
                80
            )
        )


        onset=float(
            note["onset"]
        )


        duration=float(
            note["duration"]
        )



        # WJazzD onset单位是秒
        # 需要转换成tick

        start=int(
            onset
            *
            TICKS_PER_BEAT
            /
            0.5
        )


        length=int(
            duration
            *
            TICKS_PER_BEAT
            /
            0.5
        )


        end=start+length



        events.append(

            (
                start,

                Message(
                    "note_on",
                    note=pitch,
                    velocity=velocity,
                    time=0
                )

            )

        )


        events.append(

            (
                end,

                Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    time=0
                )

            )

        )



    # 按时间排序

    events.sort(
        key=lambda x:x[0]
    )


    current_tick=0


    for tick,msg in events:


        msg.time=tick-current_tick

        track.append(msg)

        current_tick=tick



    midi.save(output_file)



    print(
        "Saved:",
        output_file
    )





if __name__=="__main__":


    json_to_midi(
        JSON_FILE,
        OUTPUT_MIDI
    )