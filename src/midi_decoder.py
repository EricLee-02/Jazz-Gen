"""
JazzGen MIDI Decoder
Decode melody tokens:
<BOS>
KEY_xxx
AVGTEMPO_xxx

SECTION_xxx
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

into MIDI.

Timing rule:
beat position =
    (TATUM - 1) / DIVISION

DIVISION means how many subdivisions exist inside the beat.
TATUM is 1-based.

Example:
DIVISION_4 + TATUM_1 -> beat + 0/4
DIVISION_4 + TATUM_2 -> beat + 1/4
DIVISION_4 + TATUM_3 -> beat + 2/4
DIVISION_4 + TATUM_4 -> beat + 3/4
"""
import mido
from mido import (
    MidiFile,
    MidiTrack,
    Message,
    MetaMessage
)
from melody_generate_model import SoloTokenConstraint

# ============================================================
# MIDI settings
# ============================================================
TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 120.0
DEFAULT_PERIOD = 4

solo_token_constraint = SoloTokenConstraint

# ============================================================
# Duration
# ============================================================

def duration_to_ticks(duration):

    """
    Convert JazzGen DURATION token into MIDI ticks.

    Current interpretation:

        DURATION_1     whole note
        DURATION_2     half note
        DURATION_4     quarter note
        DURATION_8     eighth note
        DURATION_16    sixteenth note
        DURATION_32    thirty-second note
        DURATION_LONG  >= whole-note-like long duration

    NOTE:
    This follows the symbolic meaning of the token names.
    """

    mapping = {
        "1": TICKS_PER_BEAT * 4,
        "2":TICKS_PER_BEAT * 2,
        "4":TICKS_PER_BEAT,
        "8":TICKS_PER_BEAT // 2,
        "16":TICKS_PER_BEAT // 4,
        "32":TICKS_PER_BEAT // 8,
        # Do NOT let LONG fall back to sixteenth note
        "LONG":TICKS_PER_BEAT * 4,
    }
    return mapping.get(str(duration),TICKS_PER_BEAT // 4)


# ============================================================
# Note
# ============================================================

class Note:
    def __init__(
        self,
        section,
        bar,
        period,
        beat,
        division,
        tatum,
        pitch,
        duration,
        velocity,
        articulation=None,
        micro=None
    ):
        self.section = section
        self.bar = int(bar)
        self.period = int(period)
        self.beat = int(beat)
        self.division = int(division)
        self.tatum = int(tatum)
        self.pitch = int(pitch)
        self.duration = str(duration)
        self.velocity = int(velocity)
        self.articulation = articulation
        self.micro = micro


# ============================================================
# Tempo
# ============================================================

def extract_tempo(tokens):

    """
    Read AVGTEMPO_xxx from generated token stream.

    Example:
        AVGTEMPO_120.1
    """
    for token in tokens:

        if (isinstance(token, str) and token.startswith("AVGTEMPO_")):
            try:
                return float(token.split("_", 1)[1] )

            except (ValueError, IndexError):
                pass

    return DEFAULT_TEMPO


# ============================================================
# Complete pending note safely
# ============================================================

def _finish_note( notes,pending):
    if pending is None:
        return None
    # Pitch without duration cannot safely become MIDI note.
    if pending.get("duration") is None:
        return None
    notes.append( Note(**pending))
    return None


# ============================================================
# Parse tokens
# ============================================================

def parse_notes(tokens):
    notes = []
    current = {
        "section": None,
        "bar": 0,
        "period": DEFAULT_PERIOD,
        "beat": 1,
        "division": 1,
        "tatum": 1,
    }
    pending = None
    for token in tokens:
        if not isinstance(token, str):
            continue

        # ----------------------------------------
        # EOS means previous note is complete.
        # ----------------------------------------
        if token == "<EOS>":
            pending = _finish_note(notes, pending)
            break
        if token in ["<BOS>","<PAD>","<UNK>"]:
            continue
        if "_" not in token:
            continue
        field, value = token.split( "_", 1 )

        # ====================================================
        # SECTION
        #
        # New SECTION normally means a new note event starts.
        # Flush previous pending note first.
        # ====================================================

        if field == "SECTION":
            pending = _finish_note(notes,pending)
            current["section"] = value

        # ====================================================
        # BAR
        # ====================================================

        elif field == "BAR":
            # If old sequence does not contain SECTION,
            # BAR can also mark beginning of next note.
            if pending is not None:
                pending = _finish_note(notes,pending)
            current["bar"] = int(value)

        # ====================================================
        # PERIOD
        # ====================================================
        elif field == "PERIOD":
            period = int(value)
            if period < 1:
                period = DEFAULT_PERIOD
            current["period"] = period
        # ====================================================
        # BEAT
        # ====================================================
        elif field == "BEAT":
            current["beat"] = int(value)
        # ====================================================
        # DIVISION
        # ====================================================
        elif field == "DIVISION":
            division = int(value)

            # Avoid division by zero
            current["division"] = max(division,1)
        # ====================================================
        # TATUM
        # ====================================================
        elif field == "TATUM":
            current["tatum"] = int(value)

        # ====================================================
        # PITCH
        # ====================================================
        elif field == "PITCH":
            # Defensive handling:
            # if malformed stream contains another pitch before
            # previous note was flushed, preserve previous note.
            if pending is not None:
                pending = _finish_note(notes,pending)
            pending = {
                "section": current["section"],
                "bar": current["bar"],
                "period":current["period"],
                "beat":current["beat"],
                "division":current["division"],
                "tatum":current["tatum"],
                "pitch":int(value),
                "duration": None,
                "velocity":100,
                "articulation": None,
                "micro":None,
                }
        # ====================================================
        # DURATION
        # ====================================================
        elif field == "DURATION":
            if pending is not None:
                pending["duration"] = value
        # ====================================================
        # VELOCITY
        # ====================================================
        elif field == "VELOCITY":
            if pending is not None:
                velocity = int(value)
                pending["velocity"] = max( 1, min(velocity,127))
        # ====================================================
        # ARTICULATION
        # ====================================================
        elif field == "ARTIC":
            if pending is not None:
                pending["articulation"] = (value)
        # ====================================================
        # MICRO TIMING
        # ====================================================
        elif field == "MICRO":
            if pending is not None:
                pending["micro"] = value
        # KEY / AVGTEMPO / CHORD / etc.
        # are intentionally ignored by the MIDI note parser.

    # ========================================================
    # End of stream
    # ========================================================
    pending = _finish_note( notes,pending)
    return notes


# ============================================================
# Build bar start positions
# ============================================================

def build_bar_start_ticks(notes):

    """
    Build cumulative bar-start positions.

    This supports PERIOD changes better than:

        bar * 4 * TICKS_PER_BEAT

    Example:

        bar 0 = 4/4
        bar 1 = 3/4

    bar 2 then starts after 7 beats,
    not after 8 beats.
    """
    if not notes:
        return {0: 0}
    max_bar = max(note.bar for note in notes)
    period_by_bar = {}
    for note in notes:
        if note.bar not in period_by_bar:
            period_by_bar[ note.bar] = max(note.period, 1)
    starts = { 0: 0}
    last_period = DEFAULT_PERIOD
    tick = 0
    for bar in range(max_bar + 1):
        starts[bar] = tick
        period = period_by_bar.get(bar,last_period)
        period = max(period,1)
        last_period = period
        tick += (period *TICKS_PER_BEAT)
    return starts


# ============================================================
# Micro timing
# ============================================================

def micro_offset_ticks(micro):
    if micro is None:
        return 0
    micro = str(micro).upper()
    # About 1/32 beat at PPQ=480
    amount = 15
    if "EARLY" in micro:
        return -amount
    if "LATE" in micro:
        return amount
    # MICRO_GRID
    return 0


# ============================================================
# Time conversion
# ============================================================

def note_to_tick(note, bar_start_ticks):

    """
    Correct JazzGen timing:

    absolute tick
        =
    start of BAR
        +
    (BEAT - 1) * ticks_per_beat
        +
    ((TATUM - 1) / DIVISION) * ticks_per_beat

    IMPORTANT:

    DIVISION does NOT itself advance time.

    It defines how finely the current beat is divided.
    """

    # ----------------------------------------
    # BAR
    # ----------------------------------------

    tick = bar_start_ticks.get(note.bar, note.bar *DEFAULT_PERIOD*TICKS_PER_BEAT)
    # ----------------------------------------
    # BEAT
    # ----------------------------------------
    beat = max(note.beat,1)
    tick += ( beat - 1) * TICKS_PER_BEAT
    # ----------------------------------------
    # TATUM inside beat
    # ----------------------------------------
    division = max(note.division,1)
    tatum = max(note.tatum,1)
    # Keep malformed tatum inside legal range
    tatum = min(tatum, division)
    position = solo_token_constraint.canonical_position(note.division,note.tatum)
    tick += round(float(position)*TICKS_PER_BEAT)
    # ----------------------------------------
    # MICRO timing
    # ----------------------------------------
    tick += micro_offset_ticks(note.micro)
    return max( int(tick),0)


# ============================================================
# Articulation
# ============================================================

def apply_articulation(ticks,articulation):
    ticks = max( int(ticks),1)
    if articulation is None:
        return ticks
    articulation = str(articulation).lower()
    if articulation == "staccato":
        ticks = round(ticks * 0.55)
    elif articulation == "tenuto":
        ticks = round(ticks * 1.05)
    elif articulation in [ "legato","slur"]:
        ticks = round(ticks * 1.08)
    return max(ticks,1)
# ============================================================
# MIDI export
# ============================================================

def tokens_to_midi(tokens,tempo_bpm=None):
    notes = parse_notes(tokens)
    if tempo_bpm is None:
        tempo_bpm = extract_tempo(tokens)
    if tempo_bpm <= 0:
        tempo_bpm = DEFAULT_TEMPO
    print("MIDI notes:",len(notes))
    print("MIDI tempo:",tempo_bpm)
    # ========================================================
    # MIDI
    # ========================================================
    midi = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    midi.tracks.append(track)
    # ----------------------------------------
    # Track name
    # ----------------------------------------
    track.append( MetaMessage( "track_name", name="JazzGen Solo", time=0))
    # ----------------------------------------
    # Tempo
    # ----------------------------------------
    track.append(MetaMessage( "set_tempo", tempo=mido.bpm2tempo(tempo_bpm),time=0))
    # ----------------------------------------
    # Time signature
    #
    # Most current WJazzD data is 4-period.
    # We retain the first note's PERIOD.
    # ----------------------------------------

    if notes:
        numerator = max(notes[0].period, 1)

    else:
        numerator = DEFAULT_PERIOD
    track.append( MetaMessage("time_signature", numerator=numerator,denominator=4, time=0))

    # ========================================================
    # Absolute positions
    # ========================================================
    bar_start_ticks = build_bar_start_ticks(notes)
    events = []
    for note in notes:
        start = note_to_tick(note,bar_start_ticks)
        duration = duration_to_ticks(note.duration)
        duration = apply_articulation(duration,note.articulation)
        end = ( start+duration)

        # event priority:
        #
        # note_off = 0
        # note_on  = 1
        #
        # therefore off comes first
        # when timestamps are identical.

        events.append((start, 1, "on",note))
        events.append(( end,0,"off", note))
    # ========================================================
    # Sort MIDI events
    # ========================================================
    events.sort(key=lambda x: (x[0], x[1], x[3].pitch ))

    # ========================================================
    # Convert absolute ticks -> MIDI delta ticks
    # ========================================================
    last_tick = 0
    for (tick, _,event,note) in events:
        tick = max(int(tick),last_tick)
        delta = ( tick-last_tick)
        last_tick = tick
        if event == "on":
            track.append( Message( "note_on",note=max(0,min(note.pitch,127)),velocity=max(1,min(note.velocity, 127)), time=delta))

        else:
            track.append(Message("note_off", note=max( 0,min( note.pitch,127 ) ),velocity=0,time=delta))
    return midi