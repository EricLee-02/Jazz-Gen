"""把旋律 token 转成具有相对 swing / lay-back / push 感的 MIDI。

调用仍然兼容 tokens_to_midi(tokens)，返回 mido.MidiFile。
音符顺序：BAR, BEAT, TATUM, POSITION, PITCH, DURATION, VELOCITY,
可选 ARTIC, MICRO。先读完整个音符，再计算时间，避免属性错位。

所有音乐时间先用“四分音符拍”为单位，最后才换算成整数 tick。
目前支持固定拍号 n/4；变拍号需先扩展小节到绝对拍的映射。

重要：subdivisions_per_beat 必须与实际编码器一致！默认 4 是沿用
用户原解码器对 POSITION_*-1..4 的解释，不代表已确认训练数据的网格。
出现 5/6 时会明确报错，不会映射到拍头，也不会根据出现的最大值猜网格。
它可设为整数，或函数 subdivisions_per_beat(bar, beat)，支持逐拍指定。
若源位置已包含 swing，先将 swing_ratio=1.0，避免二次 swing。
"""

from dataclasses import dataclass
from math import floor, isfinite
import warnings


TICKS_PER_BEAT = 480


@dataclass(frozen=True)
class Note:
    bar: int
    beat: int
    tatum: int
    pitch: int
    duration: str
    velocity: int
    articulation: str = None
    micro: str = None


@dataclass
class TimedNote:
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int
    bar: int
    beat: int
    tatum: int


def token_value(token):
    return token.split("_", 1)[-1]


def duration_to_beats(value):
    """沿用原解码器：8=八分音符=0.5拍，16=0.25拍，LONG=4拍。"""
    value = str(value).upper()
    if value == "LONG":
        return 4.0
    if value not in {"1", "2", "4", "8", "16", "32"}:
        raise ValueError(f"未知时值 DURATION_{value}；请按编码器定义补充映射。")
    return 4.0 / int(value)


def duration_to_tick(value, ticks_per_beat=TICKS_PER_BEAT):
    return round(duration_to_beats(value) * ticks_per_beat)


def swing_fraction(fraction, swing_ratio=1.6):
    """拍内映射：0 -> 0，0.5 -> r/(1+r)，1 -> 1。

    r=1 为直拍；r=2 为 2:1。两段分别线性插值，拍边界保持不动。
    这是可调的演奏时间模型，不是从模型权重恢复的精确演奏时值。
    """
    ratio = float(swing_ratio)
    if not isfinite(ratio) or ratio <= 0:
        raise ValueError("swing_ratio 必须是有限正数。")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction 必须在 0 到 1 之间。")
    split = ratio / (1.0 + ratio)
    if fraction <= 0.5:
        return 2.0 * fraction * split
    return split + 2.0 * (fraction - 0.5) * (1.0 - split)


def position_to_tick(position, subdivisions_per_beat=4, swing_ratio=1.6,
                     ticks_per_beat=TICKS_PER_BEAT):
    """POSITION -> 小节内 tick；此辅助函数使用固定细分数和 swing 比例。"""
    beat, tatum = map(int, position.split("-"))
    if (type(subdivisions_per_beat) is not int or subdivisions_per_beat < 1
            or beat < 1 or not 1 <= tatum <= subdivisions_per_beat):
        raise ValueError(f"POSITION_{position} 超出指定网格；请核对 subdivisions_per_beat。")
    phase = (tatum - 1) / subdivisions_per_beat
    return ((beat - 1) + swing_fraction(phase, swing_ratio)) * ticks_per_beat


def apply_micro(beat_time, micro, micro_ratio=0.03, feel_ratio=0.0):
    """输入/输出均以拍计。+0.03=晚一拍的3%，-0.03=早一拍的3%。

    feel_ratio 是整段旋律相对节拍的位移；micro 是当前音符的额外偏移。
    该函数不使用固定 tick；render_notes 会另行限制偏移，避免音符换序。
    """
    if not isfinite(micro_ratio) or not 0 <= micro_ratio < 0.5:
        raise ValueError("micro_ratio 应在 [0, 0.5) 内。")
    if not isfinite(feel_ratio) or not -0.5 < feel_ratio < 0.5:
        raise ValueError("feel_ratio 应在 (-0.5, 0.5) 内。")
    value = token_value(micro).upper() if micro else None
    if value == "EARLY":
        offset = -micro_ratio
    elif value == "LATE":
        offset = micro_ratio
    elif value in (None, "ON", "ON_TIME", "ONTIME", "NEUTRAL", "NONE"):
        offset = 0.0
    else:
        raise ValueError(f"未知 micro token: {micro!r}；请补充它的相对偏移定义。")
    return beat_time + feel_ratio + offset


def apply_articulation(duration, artic):
    """duration 可用拍或 tick；只作比例缩放，保留浮点精度。"""
    name = token_value(artic).lower() if artic else None
    if name == "staccato":
        return duration * 0.85
    if name == "legato":
        return duration * 1.10
    return duration


def parse_notes(tokens):
    """PITCH 后的 DURATION/VELOCITY/ARTIC/MICRO 归属于当前音符。

    BAR 可沿用；每个音符必须提供新的 BEAT/TATUM/POSITION。
    不修复矛盾、重复或回退的位置，避免掩盖生成端的错误。
    """
    notes = []
    time = {"BAR": None, "BEAT": None, "TATUM": None, "POSITION": None}
    pending = None

    def flush():
        nonlocal pending
        if pending is None:
            return
        if pending["duration"] is None or pending["velocity"] is None:
            raise ValueError(f"第 {len(notes) + 1} 个音符不完整，缺 DURATION 或 VELOCITY。")
        duration_to_beats(pending["duration"])
        if not 0 <= pending["pitch"] <= 127 or not 0 <= pending["velocity"] <= 127:
            raise ValueError("PITCH / VELOCITY 必须位于 MIDI 的 0..127 范围。")
        notes.append(Note(**pending))
        pending = None
        for key in ("BEAT", "TATUM", "POSITION"):
            time[key] = None

    tail_fields = {"DURATION": "duration", "VELOCITY": "velocity",
                   "ARTIC": "articulation", "MICRO": "micro"}
    for index, token in enumerate(tokens):
        if not isinstance(token, str):
            raise TypeError(f"token {index} 必须是字符串。")
        if token in ("<BOS>", "<PAD>"):
            continue
        if token == "<EOS>":
            flush()
            break
        field, _, value = token.partition("_")
        if field in time:
            flush()
            if field == "BAR":
                for key in ("BEAT", "TATUM", "POSITION"):
                    time[key] = None
            time[field] = value if field == "POSITION" else int(value)
        elif field == "PITCH":
            flush()
            if any(value is None for value in time.values()):
                raise ValueError(f"token {index}: PITCH 前缺完整 BAR/BEAT/TATUM/POSITION。")
            expected = f"{time['BEAT']}-{time['TATUM']}"
            if time["POSITION"] != expected:
                raise ValueError(f"token {index}: POSITION_{time['POSITION']} 与 "
                                 f"BEAT/TATUM {expected} 矛盾，请先修正生成结果。")
            pending = dict(bar=time["BAR"], beat=time["BEAT"], tatum=time["TATUM"],
                           pitch=int(value), duration=None, velocity=None,
                           articulation=None, micro=None)
        elif field in tail_fields:
            if pending is None:
                raise ValueError(f"token {index}: {token} 前没有所属 PITCH。")
            key = tail_fields[field]
            if pending[key] is not None:
                raise ValueError(f"token {index}: 同一音符重复出现 {field}。")
            pending[key] = int(value) if field == "VELOCITY" else (
                token if field in ("ARTIC", "MICRO") else value)
        # 和弦、段落等非音符 token 不用于此旋律轨的音符时间计算。
    flush()
    if any(time[key] is not None for key in ("BEAT", "TATUM", "POSITION")):
        raise ValueError("序列结尾有未完成的音符位置字段。")
    return notes


def render_notes(notes, *, ticks_per_beat=TICKS_PER_BEAT, beats_per_bar=4,
                 bar_base=0, subdivisions_per_beat=4, swing_ratio=1.6,
                 feel_ratio=0.0, micro_ratio=0.03, monophonic=True):
    """Note -> TimedNote；所有节奏比例在最后转 tick 前计算。

    swing_ratio 可为数字或函数 swing_ratio(bar, beat)。相同拍只求值一次。
    bar_base=0 沿用原式 bar*4；若编码从 BAR_1 起且应从零秒开始，设为1。
    monophonic=True 把前一音的 note_off 限制在下一音 note_on 处。
    起音仍须严格递增；重复位置不会被静默删除或改成别的位置。
    """
    for name, value in (("ticks_per_beat", ticks_per_beat), ("beats_per_bar", beats_per_bar)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} 必须是正整数。")
    if type(bar_base) is not int or bar_base < 0:
        raise ValueError("bar_base 必须是非负整数。")
    apply_micro(0, None, micro_ratio, feel_ratio)  # Validate even for an empty stream.
    notes = list(notes)
    ratio_cache, divisions_cache = {}, {}

    def ratio_at(bar, beat):
        key = (bar, beat)
        if key not in ratio_cache:
            ratio = swing_ratio(bar, beat) if callable(swing_ratio) else swing_ratio
            swing_fraction(0.5, ratio)  # Validate the ratio.
            ratio_cache[key] = float(ratio)
        return ratio_cache[key]

    def warp(absolute_beat):
        whole = floor(absolute_beat)
        bar = whole // beats_per_bar + bar_base
        beat = whole % beats_per_bar + 1
        return whole + swing_fraction(absolute_beat - whole, ratio_at(bar, beat))

    nominal, swung, spans = [], [], []
    for note in notes:
        if note.bar < bar_base or not 1 <= note.beat <= beats_per_bar:
            raise ValueError(f"无效小节/拍号: BAR_{note.bar}, BEAT_{note.beat}。")
        key = (note.bar, note.beat)
        if key not in divisions_cache:
            count = subdivisions_per_beat(*key) if callable(subdivisions_per_beat) else subdivisions_per_beat
            if type(count) is not int or count < 1:
                raise ValueError("subdivisions_per_beat 必须返回正整数。")
            divisions_cache[key] = count
        count = divisions_cache[key]
        if not 1 <= note.tatum <= count:
            raise ValueError(f"BAR_{note.bar}, POSITION_{note.beat}-{note.tatum} 超出每拍 {count} 格。"
                             "请核对编码器的真实细分数；不能由最大 TATUM 自动推断。")
        onset = (note.bar - bar_base) * beats_per_bar + note.beat - 1 + (note.tatum - 1) / count
        if nominal and onset <= nominal[-1]:
            raise ValueError(f"第 {len(nominal) + 1} 个音符重复起音或时间回退，请先修正生成结果。")
        nominal.append(onset)
        base = warp(onset)
        swung.append(base)
        # Transform both endpoints so swung eighth-note lengths follow the grid.
        spans.append(apply_articulation(
            warp(onset + duration_to_beats(note.duration)) - base, note.articulation))

    result = []
    limited_micro = clipped_start = rounded_order = 0
    for i, note in enumerate(notes):
        micro_offset = apply_micro(0.0, note.micro, micro_ratio, 0.0)
        gaps = []
        if i:
            gaps.append(swung[i] - swung[i - 1])
        if i + 1 < len(notes):
            gaps.append(swung[i + 1] - swung[i])
        if gaps:
            # Two adjacent notes can each move at most 45% of their gap.
            # Their order therefore survives different EARLY/LATE labels.
            limit = 0.45 * min(gaps)
            bounded = max(-limit, min(limit, micro_offset))
            limited_micro += abs(bounded - micro_offset) > 1e-12
            micro_offset = bounded
        start_beats = swung[i] + feel_ratio + micro_offset
        clipped_start += start_beats < 0
        start_tick = max(0, round(start_beats * ticks_per_beat))
        if result and start_tick <= result[-1].start_tick:
            # Only MIDI integer rounding/file-start clipping is handled here.
            # This is not a musical timing offset in the swing algorithm.
            start_tick = result[-1].start_tick + 1
            rounded_order += 1
        end_tick = start_tick + max(1, round(spans[i] * ticks_per_beat))
        result.append(TimedNote(note.pitch, note.velocity, start_tick, end_tick,
                                note.bar, note.beat, note.tatum))
    if monophonic:
        for previous, following in zip(result, result[1:]):
            previous.end_tick = min(previous.end_tick, following.start_tick)
    if limited_micro or clipped_start or rounded_order:
        warnings.warn(f"时间边界处理：{limited_micro} 个 micro 偏移受相邻音间距限制；"
                      f"{clipped_start} 个起音早于文件零点；{rounded_order} 个起音作整数 tick 顺序调整。",
                      RuntimeWarning, stacklevel=2)
    return result


def tokens_to_midi(tokens, *, tempo_bpm=120, ticks_per_beat=TICKS_PER_BEAT,
                   beats_per_bar=4, bar_base=0, subdivisions_per_beat=16,
                   swing_ratio=1.6, feel_ratio=0.0, micro_ratio=0.03,
                   monophonic=True):
    """例：tokens_to_midi(tokens, swing_ratio=1.6, feel_ratio=0.03)。

    +feel_ratio=lay back；-feel_ratio=push。整轨位移需要相对伴奏/节拍器听。
    swing_ratio 调整拍内比例；MICRO token 给各音符增加拍长比例的偏移。
    mido 只在真正导出时导入，时间映射函数可独立检查。
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo

    if not isfinite(tempo_bpm) or tempo_bpm <= 0:
        raise ValueError("tempo_bpm 必须为有限正数。")
    tempo = bpm2tempo(tempo_bpm)
    if not 1 <= tempo <= 0xFFFFFF:
        raise ValueError("tempo_bpm 超出 MIDI tempo 可表示范围。")
    if not 1 <= beats_per_bar <= 255:
        raise ValueError("beats_per_bar 超出 MIDI 拍号范围。")
    notes = render_notes(
        parse_notes(tokens), ticks_per_beat=ticks_per_beat, beats_per_bar=beats_per_bar,
        bar_base=bar_base, subdivisions_per_beat=subdivisions_per_beat,
        swing_ratio=swing_ratio, feel_ratio=feel_ratio, micro_ratio=micro_ratio,
        monophonic=monophonic,
    )
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(MetaMessage("time_signature", numerator=beats_per_bar, denominator=4, time=0))
    events = []
    for note in notes:
        events.append((note.start_tick, 1, note.pitch, note.velocity))
        events.append((note.end_tick, 0, note.pitch, 0))
    events.sort(key=lambda event: (event[0], event[1]))  # 同 tick 先 off，后 on。
    last_tick = 0
    for tick, is_on, pitch, velocity in events:
        track.append(Message("note_on" if is_on else "note_off", note=pitch,
                             velocity=velocity, time=tick - last_tick))
        last_tick = tick
    track.append(MetaMessage("end_of_track", time=0))
    return mid
