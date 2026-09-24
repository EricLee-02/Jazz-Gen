from pathlib import Path
from collections import defaultdict, Counter
from fractions import Fraction
import json
import statistics


# ============================================================
# 修改成你的 456 首 JSON 所在目录
# ============================================================

JSON_DIR = Path(
    '/Volumes/My Passport/Jazz Gen/data/processed/jazz_json_v2_token'
)

GRID_PER_BEAT = 24
BEATS_PER_BAR = 4


# ============================================================
# Duration -> beat length
# ============================================================

DURATION_TO_BEATS = {
    "DURATION_2": 2.0,
    "DURATION_4": 1.0,
    "DURATION_8": 0.5,
    "DURATION_16": 0.25,
    "DURATION_32": 0.125,
}


# ============================================================
# 与你 generation 当前 canonical_position 保持一致
# ============================================================

def canonical_position(
    division,
    tatum
):
    division = int(division)
    tatum = int(tatum)

    if division < 1:
        return None

    if tatum < 1:
        return None

    if tatum > division:
        return None

    raw_position = Fraction(
        tatum - 1,
        division
    )

    grid_index = round(
        float(raw_position)
        * GRID_PER_BEAT
    )

    grid_index = max(
        0,
        min(
            GRID_PER_BEAT - 1,
            grid_index
        )
    )

    return float(
        Fraction(
            grid_index,
            GRID_PER_BEAT
        )
    )


# ============================================================
# 递归提取 JSON 中所有 token
#
# 你的 JSON 前面已经能扫描到 DURATION_x，
# 所以这里不依赖固定 JSON 层级。
# ============================================================

def collect_tokens(obj, result):

    if isinstance(obj, str):

        prefixes = (
            "SECTION_",
            "BAR_",
            "PERIOD_",
            "BEAT_",
            "DIVISION_",
            "TATUM_",
            "PITCH_",
            "DURATION_",
            "VELOCITY_",
        )

        if obj.startswith(prefixes):
            result.append(obj)

        return


    if isinstance(obj, list):

        for item in obj:
            collect_tokens(
                item,
                result
            )

        return


    if isinstance(obj, dict):

        for value in obj.values():
            collect_tokens(
                value,
                result
            )

        return


# ============================================================
# token -> note records
# ============================================================

def parse_notes(tokens):

    notes = []

    current = {}

    required = (
        "BAR",
        "BEAT",
        "DIVISION",
        "TATUM",
        "PITCH",
        "DURATION",
    )


    for token in tokens:

        if "_" not in token:
            continue

        field, value = token.split(
            "_",
            1
        )


        if field == "BAR":
            current["BAR"] = value

        elif field == "PERIOD":
            current["PERIOD"] = value

        elif field == "BEAT":
            current["BEAT"] = value

        elif field == "DIVISION":
            current["DIVISION"] = value

        elif field == "TATUM":
            current["TATUM"] = value

        elif field == "PITCH":
            current["PITCH"] = value

        elif field == "DURATION":
            current["DURATION"] = (
                "DURATION_" + value
            )

        elif field == "VELOCITY":

            current["VELOCITY"] = value

            if all(
                x in current
                for x in required
            ):

                try:

                    bar = int(
                        current["BAR"]
                    )

                    beat = int(
                        current["BEAT"]
                    )

                    division = int(
                        current["DIVISION"]
                    )

                    tatum = int(
                        current["TATUM"]
                    )

                    position = (
                        canonical_position(
                            division,
                            tatum
                        )
                    )

                    if position is None:
                        current = {}
                        continue


                    absolute_onset = (
                        bar * BEATS_PER_BAR
                        + beat - 1
                        + position
                    )


                    notes.append({
                        "bar": bar,
                        "beat": beat,
                        "division": division,
                        "tatum": tatum,
                        "pitch": int(
                            current["PITCH"]
                        ),
                        "duration_token":
                            current["DURATION"],
                        "onset":
                            absolute_onset,
                    })

                except Exception:
                    pass


            current = {}

    return notes


# ============================================================
# Main analysis
# ============================================================

files = list(
    JSON_DIR.rglob("*.json")
)

print(
    "JSON files:",
    len(files)
)


all_records = []

unknown_duration = Counter()

total_notes = 0


for file in files:

    try:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        tokens = []

        collect_tokens(
            data,
            tokens
        )


        notes = parse_notes(
            tokens
        )

        total_notes += len(notes)


        # ----------------------------------------
        # IOI 必须在同一首歌内部计算
        # ----------------------------------------

        for i in range(
            len(notes) - 1
        ):

            current = notes[i]
            next_note = notes[i + 1]

            ioi = (
                next_note["onset"]
                - current["onset"]
            )


            # 非法/重复/倒退 onset
            if ioi <= 0:
                continue


            duration_token = (
                current["duration_token"]
            )


            if (
                duration_token
                not in DURATION_TO_BEATS
            ):

                unknown_duration[
                    duration_token
                ] += 1

                continue


            duration_beats = (
                DURATION_TO_BEATS[
                    duration_token
                ]
            )


            gate_ratio = (
                duration_beats
                / ioi
            )


            all_records.append({
                "duration":
                    duration_token,
                "duration_beats":
                    duration_beats,
                "ioi":
                    ioi,
                "gate_ratio":
                    gate_ratio,
            })


    except Exception as e:

        print(
            "Error:",
            file.name,
            e
        )


print()
print(
    "Parsed notes:",
    total_notes
)

print(
    "Valid IOI records:",
    len(all_records)
)

print(
    "Unknown duration:",
    unknown_duration
)
# ============================================================
# Statistics by Duration
# ============================================================

groups = defaultdict(list)

for record in all_records:

    groups[
        record["duration"]
    ].append(record)


duration_order = [
    "DURATION_2",
    "DURATION_4",
    "DURATION_8",
    "DURATION_16",
    "DURATION_32",
]


print()
print("=" * 105)

print(
    f"{'Duration':15s}"
    f"{'Count':>10s}"
    f"{'Mean IOI':>12s}"
    f"{'Median IOI':>14s}"
    f"{'Median Gate':>14s}"
    f"{'<0.5':>10s}"
    f"{'0.5-0.8':>12s}"
    f"{'0.8-1.05':>12s}"
    f"{'>1.05':>10s}"
)

print("=" * 105)


for duration in duration_order:

    rows = groups.get(
        duration,
        []
    )

    if not rows:
        continue


    iois = [
        r["ioi"]
        for r in rows
    ]

    gates = [
        r["gate_ratio"]
        for r in rows
    ]


    n = len(rows)


    very_short = (
        sum(
            g < 0.5
            for g in gates
        )
        / n * 100
    )


    short = (
        sum(
            0.5 <= g < 0.8
            for g in gates
        )
        / n * 100
    )


    connected = (
        sum(
            0.8 <= g <= 1.05
            for g in gates
        )
        / n * 100
    )


    overlap = (
        sum(
            g > 1.05
            for g in gates
        )
        / n * 100
    )


    print(
        f"{duration:15s}"
        f"{n:10d}"
        f"{statistics.mean(iois):12.3f}"
        f"{statistics.median(iois):14.3f}"
        f"{statistics.median(gates):14.3f}"
        f"{very_short:10.2f}"
        f"{short:12.2f}"
        f"{connected:12.2f}"
        f"{overlap:10.2f}"
    )


print("=" * 105)
all_gates = [
    r["gate_ratio"]
    for r in all_records
]

all_iois = [
    r["ioi"]
    for r in all_records
]


print()
print("===== OVERALL =====")

print(
    "Median IOI:",
    round(
        statistics.median(
            all_iois
        ),
        3
    )
)

print(
    "Median Gate Ratio:",
    round(
        statistics.median(
            all_gates
        ),
        3
    )
)

print(
    "Mean Gate Ratio:",
    round(
        statistics.mean(
            all_gates
        ),
        3
    )
)


n = len(all_gates)

print(
    "Gate < 0.5:",
    round(
        sum(
            g < 0.5
            for g in all_gates
        )
        / n * 100,
        2
    ),
    "%"
)

print(
    "Gate 0.5-0.8:",
    round(
        sum(
            0.5 <= g < 0.8
            for g in all_gates
        )
        / n * 100,
        2
    ),
    "%"
)

print(
    "Gate 0.8-1.05:",
    round(
        sum(
            0.8 <= g <= 1.05
            for g in all_gates
        )
        / n * 100,
        2
    ),
    "%"
)

print(
    "Gate > 1.05:",
    round(
        sum(
            g > 1.05
            for g in all_gates
        )
        / n * 100,
        2
    ),
    "%"
)