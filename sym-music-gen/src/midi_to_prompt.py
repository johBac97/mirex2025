import symusic
import json
from pathlib import Path
import argparse


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("midifile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-bars", type=int)

    return parser.parse_args()


def midi_to_json(midi_file: Path, output_file: Path, max_bars: int = 16):
    score = symusic.Score(midi_file).to("quarter")

    json_notes = []
    for note in score.tracks[0].notes:
        if max_bars and round(note.time) >= max_bars * 4:
            break

        # Symusic has the duration and start in quarter notes. Cast it to closest sixteenth
        start = round(note.time * 4)
        duration = round(note.duration * 4)
        json_notes.append({"start": start, "duration": duration, "pitch": note.pitch})

    with output_file.open("w") as io:
        json.dump(json_notes, io, indent=4)


def main():
    args = __parse_args()

    midi_to_json(args.midifile, args.output, args.max_bars)


if __name__ == "__main__":
    main()
