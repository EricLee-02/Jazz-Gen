# """Harmony-conditioned solo generation with constraints applied before sampling.

# This uses the note format in your uploaded output:
# BAR,PERIOD, BEAT, DIVISION TATUM, PITCH, DURATION, VELOCITY, [ARTIC], [MICRO].
# ARTIC/MICRO are included if those token families exist in the vocabulary.
# BEAT and TATUM are 1-based. TATUM values come from the vocabulary, not an
# assumed four subdivisions per beat. No duration-unit conversion is assumed.

# POSITION has been removed because its information is already represented by
# BEAT + DIVISION + TATUM.

# The neural layers and checkpoint parameter names are unchanged.
# """

# import re
# from collections import Counter

# import torch
# import torch.nn as nn
# from fractions import Fraction

# """Keep complete note onsets strictly increasing in (bar, beat, tatum).

# This intentionally emits melody note events only. Harmony is supplied to
# the encoder; harmony/header tokens are not emitted by this note grammar.
# It prevents equal onsets, not duration overlap between successive notes.
# """

# class SoloTokenConstraint:

#     def __init__(
#         self,
#         token_to_id,
#         eos_id=None,
#         beats_per_bar=None,
#         start_bar=None,
#         max_bar=None
#     ):

#         self.vocab = dict(token_to_id)
#         self.tokens = {i:t for t,i in self.vocab.items()}
#         self.eos_id=eos_id
#         self.beats_per_bar=beats_per_bar
#         self.start_bar=start_bar
#         self.max_bar=max_bar
#         self.fields=[
#             "SECTION",
#             "BAR",
#             "PERIOD",
#             "BEAT",
#             "DIVISION",
#             "TATUM",
#             "PITCH",
#             "DURATION",
#             "VELOCITY"
#         ]
#         self.groups={}
#         for field in self.fields + ["ARTIC","MICRO"]:
#             self.groups[field]=[idx for token,idx in self.vocab.items() if token.startswith( field+"_")]
#         for field in self.fields:
#             if not self.groups[field]:
#                 raise ValueError( f"Missing {field} vocabulary")

#         if self.groups["ARTIC"]:
#             self.fields.append( "ARTIC")
#         if self.groups["MICRO"]:
#             self.fields.append( "MICRO")
#         self.bars=self._numeric_tokens("BAR")
#         self.periods=self._numeric_tokens("PERIOD")
#         self.beats=self._numeric_tokens("BEAT")
#         self.divisions=self._numeric_tokens("DIVISION")
#         self.tatums=self._numeric_tokens("TATUM")


#         if max_bar is not None:
#             self.bars={ k:v for k,v in self.bars.items() if k<=max_bar}
#         self.stage=0
#         self.section = None
#         self.bar=None
#         self.period=None
#         self.beat=None
#         self.division=None
#         self.tatum=None
#         self.last_onset=None
#         self.note_count=0
#         self.finished=False




#     def _numeric_tokens(self,prefix):
#         result={}
#         for token,idx in self.vocab.items():
#             if token.startswith(prefix+"_"):
#                 try:
#                     value=int( token.split("_")[1])
#                     result[value]=idx
#                 except:
#                     pass
#         return result



#     def _position(self,division,tatum):

#         return Fraction(tatum-1, division)



#     def _future(self,bar,beat,division,tatum):
#         if self.last_onset is None:
#             return True
#         return (bar,beat,self._position(division,tatum)) > self.last_onset



#     def allowed_ids(self):
#         if self.finished:
#             return []
#         field=self.fields[self.stage]
#         if field == "SECTION":
#             ids = list(self.groups["SECTION"])
#             if (
#                 self.eos_id is not None 
#                 and self.last_onset is not  None 
#                 and self.max_bar is not None 
#                 and self.last_onset[0]>= self.max_bar):
#                 ids.append(self.eos_id)
#             return ids
        
#         if field == "BAR":
#             ids = []
#             for bar , idx in self.bars.items():
#                 if self.last_onset is None:
#                     if (self.start_bar is None
#                         or bar == self.start_bar):
#                         ids.append(idx)
#                     continue
#                 last_bar = self.last_onset[0]
#                 if bar < last_bar:
#                     continue
#                 ids.append(idx)
#             return ids
#         # =========================
#         # PERIOD
#         # =========================
#         if field=="PERIOD":
#             ids = []
#             for period, idx in self.periods.items():
#                 if period < 1:
#                     continue
#                 if self.last_onset is not None:
#                     last_bar = self.last_onset[0]
#                     last_beat = self.last_onset[1]
#                     if self.bar == last_bar:
#                         if period < last_beat:
#                             continue
#                 ids.append(idx)
#             return ids
#         # =========================
#         # BEAT
#         # =========================
#         if field=="BEAT":
#             ids=[]
#             for beat,idx in self.beats.items():
#                 if beat<1:
#                     continue
#                 if self.last_onset is not None:
#                     last_bar = self.last_onset[0]
#                     last_beat = self.last_onset[1]
#                     if self.bar == last_bar:
#                         if beat < last_beat:
#                             continue
#                 ids.append(idx)
#             return ids
#         # =========================
#         # DIVISION
#         # =========================
#         if field=="DIVISION":
#             ids = []
#             for division, idx in self.divisions.items():
#                 if division < 1:
#                     continue
#                 if self.last_onset is None:
#                     ids.append(idx)
#                     continue
#                 last_bar = self.last_onset[0]
#                 last_beat = self.last_onset[1]
#                 last_position = self.last_onset[2]
#                 if self.bar > last_bar:
#                     ids.append(idx)
#                     continue
#                 if self.bar == last_bar and self.beat>last_beat:
#                     ids.append(idx)
#                     continue
#                 if self.bar == last_bar and self.beat == last_beat:
#                     has_future_tatum = False
#                     for tatum in self.tatums.keys():
#                         if tatum<1:
#                             continue
#                         if tatum > division:
#                             continue
#                         position = self._position(division,tatum)
#                         if position > last_position:
#                             has_future_tatum = True
#                             break
#                     if not has_future_tatum:
#                         continue
#                     ids.append(idx)
#             return ids
#         # =========================
#         # TATUM
#         # =========================
#         if field=="TATUM":
#             ids=[]
#             for tatum,idx in self.tatums.items():
#                 if tatum < 1:
#                     continue
#                 if tatum > self.division:
#                     continue
#                 if self._future(self.bar,self.beat,self.division,tatum):
#                     ids.append(idx)
#             return ids
#         # =========================
#         # Melody tokens
#         # =========================

#         # Fully open:
#         # let Transformer decide
#         return self.groups[field]



#     def advance(self,token_id):
#         if token_id==self.eos_id:
#             self.finished=True
#             return
#         field=self.fields[self.stage]
#         token=self.tokens[token_id]
#         if field == "SECTION":
#             self.section = token.split("_",1)[1]
#         if field in [
#             "BAR",
#             "PERIOD",
#             "BEAT",
#             "DIVISION",
#             "TATUM"
#             ]:
#             value=int(token.split("_")[1])
#             setattr( self,field.lower(), value)
#         self.stage+=1
#         if self.stage==len(self.fields):
#             onset=(self.bar,self.beat,self._position(self.division,self.tatum))
#             self.last_onset=onset
#             self.note_count+=1
#             self.stage=0

# def audit_melody_tokens(tokens):
#     """Read note onsets independently of the sampling state machine."""
#     current = {}
#     onsets = []
#     invalid_grid = 0 
#     missing = 0
#     time_fileds = (
#         "BAR",
#         "PERIOD",
#         "BEAT",
#         "DIVISION",
#         "TATUM"
#     )
#     for token in tokens:
#         if "_" not in token:
#             continue
#         field, value = token.split("_", 1)
#         if field in time_fileds:
#             current[field] = value
#         elif field == "PITCH":
#             if any(field  not in current for field in time_fileds):
#                 missing += 1
#                 current = {}
#                 continue
#             bar = int(current["BAR"])
#             period = int(current["PERIOD"])
#             beat = int(current["BEAT"])
#             division = int(current["DIVISION"])
#             tatum = int(current["TATUM"])
#             if(period < 1 or beat < 1 or beat > period or division < 1 or tatum < 1 or tatum > division ):
#                 invalid_grid += 1
#             position = Fraction(tatum-1, division)
#             onsets.append((bar, beat, position))
#             current = {}
#     counts = Counter(onsets)
#     return {
#         "note_count": len(onsets),
#         "missing_time_fields": missing,
#         "position_mismatches": invalid_grid,
#         "duplicate_onset_groups": sum(n > 1 for n in counts.values()),
#         "extra_notes_at_same_onset": sum(n - 1 for n in counts.values()),
#         "time_backward_steps": sum(b < a for a, b in zip(onsets, onsets[1:])),
#     }


# class JazzGenerationModel(nn.Module):
#     def __init__(self, harmony_encoder, melody_decoder):
#         super().__init__()
#         self.harmony_encoder = harmony_encoder
#         for p in self.harmony_encoder.parameters():
#             p.requires_grad = False
#         self.melody_decoder = melody_decoder

#     def forward(self, harmony, melody_input):
#         memory = self.harmony_encoder.encode(**harmony)
#         return self.melody_decoder(melody_input, memory)

#     @torch.no_grad()
#     def encode_harmony(self, harmony):
#         self.eval()
#         return self.harmony_encoder.encode(**harmony)

#     @torch.no_grad()
#     def generate(
#         self,
#         harmony,
#         prompt_ids,
#         eos_id=None,
#         max_length=2048,
#         temperature=1.0,
#         top_k=80,
#         *,
#         token_to_id=None,
#         beats_per_bar=None,
#         start_bar=None,
#         max_bar=None,
#         seed=None,
#         return_info=False,
#         chords = None,
#         harmony_generator = None,
#         dynamic_harmony = False
#         ):


#         if token_to_id is None:
#             raise ValueError("token_to_id required")
#         if prompt_ids.ndim != 2:
#             raise ValueError("prompt_ids must be [1,length]")
#         bos_id = token_to_id["<BOS>"]
#         if prompt_ids[0,0].item()!=bos_id:
#             raise ValueError( "Prompt must start with BOS" )
#         constraint = SoloTokenConstraint(
#         token_to_id,
#         eos_id,
#         beats_per_bar,
#         start_bar,
#         max_bar
#         )
#         self.eval()

#         memory = self.harmony_encoder.encode(**harmony)
#         note_chords = []
#         generated = prompt_ids.to(memory.device)
#         rng=None
#         if seed is not None:
#             rng=torch.Generator( device=memory.device)
#             rng.manual_seed(seed)
#         stop_reason="token_budget"
#         while generated.size(1)<max_length:
#         # -------------------------
#         # prevent incomplete note
#         # -------------------------
#             if constraint.stage==0:
#                 remain=max_length-generated.size(1)
#                 if remain < len(constraint.fields)+1:
#                     break
#             allowed=constraint.allowed_ids()


#             if not allowed:
#                 stop_reason="no_future_position"
#                 break



#         # =========================
#         # keep musical condition
#         # =========================
#             max_ctx=self.melody_decoder.max_seq_len
#             if generated.size(1)>max_ctx:
#                 decoder_input=torch.cat([ generated[:,:3],generated[:,-(max_ctx-3):]],dim=1)
#             else:
#                 decoder_input=generated
#             logits=self.melody_decoder(decoder_input, memory)
#             scores=logits[0,-1].clone()
#             candidate_ids=torch.tensor(allowed, device=scores.device )
#             scores=scores[candidate_ids]
#             finite=torch.isfinite( scores)
#             candidate_ids=candidate_ids[finite]
#             scores=scores[finite]

#             if scores.numel()==0:
#                 raise RuntimeError( "No legal token" )

#         # =========================
#         # sampling
#         # =========================
#             if temperature==0:
#                 next_id=candidate_ids[ scores.argmax()].item()

#             else:
#                 scores=scores/temperature
#                 if top_k>0:
#                     k=min(top_k,scores.numel())
#                     scores,idx=torch.topk(scores,k)
#                     candidate_ids=candidate_ids[idx]
#                 probs=torch.softmax(scores,dim=-1)
#                 choice=torch.multinomial(probs,1,generator=rng )
#                 next_id=candidate_ids[choice].item()
#             next_token = constraint.tokens[next_id]
#             constraint.advance(next_id)
#             generated=torch.cat([generated, generated.new_tensor([[next_id]] )],dim=1)
# # ==========================================
# # Dynamic note-to-chord conditioning
# #
# # 一旦当前note的BAR已经确定，
# # 就能知道这个note属于哪个chord。
# #
# # 之后 PERIOD / BEAT / DIVISION /
# # TATUM / PITCH 等都会使用更新后的
# # harmony memory。
# # ==========================================
#             if (dynamic_harmony 
#                 and chords is not None 
#                 and harmony_generator is not None
#                 and next_token.startswith("BAR_")
#                 ):
#                 current_bar = int(next_token.split("_",1)[1])
#                 currrent_chord = harmony_generator.chord_for_bar(chords,current_bar)
#                 note_chords.append(currrent_chord)
#                 note_harmony = harmony_generator.build_note_aligned(note_chords)
#                 note_harmony = {k:v.to(generated.device) for k, v in note_harmony.items()}
#                 memory = self.harmony_encoder.encode(**note_harmony)
#             if constraint.finished:
#                 stop_reason="model_eos"
#                 break

#     # =========================
#     # force EOS safely
#     # =========================
#         forced_eos=False
#         if eos_id is not None and not constraint.finished:
#             if constraint.stage!=0:
#             # discard incomplete event
#                 while (constraint.stage!=0 and generated.size(1)>0):
#                     generated=generated[:,:-1]
#                     break
#             constraint.advance(eos_id)
#             generated=torch.cat([ generated, generated.new_tensor( [[eos_id]] )],dim=1)
#             forced_eos=True
#         info={
#             "stop_reason": stop_reason,
#             "forced_eos": forced_eos,
#             "note_count": constraint.note_count,
#             "note_harmony_count":len(note_chords),
#             "dynamic_harmony":dynamic_harmony,
#             "token_count":generated.size(1),
#             "temperature":temperature,
#             "top_k":top_k,
#             "seed":seed,
#             "max_bar":max_bar
#         }
#         return ( generated, info) if return_info else generated

"""Harmony-conditioned solo generation with constraints applied before sampling.

This uses the note format in your uploaded output:
BAR,PERIOD, BEAT, DIVISION TATUM, PITCH, DURATION, VELOCITY, [ARTIC], [MICRO].
ARTIC/MICRO are included if those token families exist in the vocabulary.
BEAT and TATUM are 1-based. TATUM values come from the vocabulary, not an
assumed four subdivisions per beat. No duration-unit conversion is assumed.

POSITION has been removed because its information is already represented by
BEAT + DIVISION + TATUM.

The neural layers and checkpoint parameter names are unchanged.
"""

import re
from collections import Counter

import torch
import torch.nn as nn
from fractions import Fraction

"""Keep note onsets increasing by bar, beat and exact tatum fraction.

SECTION may change only between bars. PERIOD stays fixed unless a source
period_by_bar map is provided. A source section_by_bar map can lock the form.
The grammar emits notes only; it cannot represent an empty bar or stop notes
from overlapping when their DURATION values exceed the onset spacing.
"""

class SoloTokenConstraint:

    def __init__(
        self,
        token_to_id,
        eos_id=None,
        beats_per_bar=None,
        start_bar=None,
        max_bar=None,
        section_by_bar=None,
        period_by_bar=None,
        max_division=None,
    ):
        self.vocab = dict(token_to_id)
        self.tokens = {i: t for t, i in self.vocab.items()}
        if eos_id is not None and self.tokens.get(eos_id) != "<EOS>":
            raise ValueError("eos_id must identify <EOS> in token_to_id")
        if beats_per_bar is not None and beats_per_bar < 1:
            raise ValueError("beats_per_bar must be positive")
        if max_division is not None and max_division < 1:
            raise ValueError("max_division must be positive")
        self.eos_id = eos_id
        self.beats_per_bar = beats_per_bar
        self.max_division = max_division
        self.fields = [
            "SECTION", "BAR", "PERIOD", "BEAT", "DIVISION", "TATUM",
            "PITCH", "DURATION", "VELOCITY"
        ]
        self.groups = {
            field: [idx for token, idx in self.vocab.items()
                    if token.startswith(field + "_")]
            for field in self.fields + ["ARTIC", "MICRO"]
        }
        for field in self.fields:
            if not self.groups[field]:
                raise ValueError(f"Missing {field} vocabulary")
        for field in ("ARTIC", "MICRO"):
            if self.groups[field]:
                self.fields.append(field)

        self.bars = self._numeric_tokens("BAR")
        self.periods = self._numeric_tokens("PERIOD")
        self.beats = self._numeric_tokens("BEAT")
        self.divisions = self._numeric_tokens("DIVISION")
        self.tatums = self._numeric_tokens("TATUM")
        if beats_per_bar is not None and beats_per_bar not in self.periods:
            raise ValueError(f"PERIOD_{beats_per_bar} is unavailable")
        self.section_by_bar = None
        if section_by_bar is not None:
            self.section_by_bar = {}
            for bar, section in section_by_bar.items():
                name = str(section)
                token = name if name.startswith("SECTION_") else "SECTION_" + name
                if token not in self.vocab:
                    raise ValueError(f"Unknown section token for bar {bar}: {token}")
                self.section_by_bar[int(bar)] = token
        self.period_by_bar = None
        if period_by_bar is not None:
            self.period_by_bar = {int(bar): int(period)
                                  for bar, period in period_by_bar.items()}
            for bar, period in self.period_by_bar.items():
                if period not in self.periods or period < 1:
                    raise ValueError(f"Unknown PERIOD_{period} for bar {bar}")

        if not self.bars:
            raise ValueError("No numeric BAR tokens in vocabulary")
        self.start_bar = min(self.bars) if start_bar is None else start_bar
        self.max_bar = max_bar
        if max_bar is not None:
            self.bars = {bar: idx for bar, idx in self.bars.items()
                         if bar <= max_bar}
        if self.start_bar not in self.bars:
            raise ValueError(f"BAR_{self.start_bar} is unavailable")
        self.grids = [
            (division, tatum, self._position(division, tatum))
            for division in self.divisions
            if division >= 1 and (max_division is None or division <= max_division)
            for tatum in self.tatums
            if 1 <= tatum <= division
        ]
        if not self.grids:
            raise ValueError("No valid DIVISION/TATUM pairs")

        self.stage = 0
        self.section = None
        self.bar = None
        self.period = None
        self.beat = None
        self.division = None
        self.tatum = None
        self.last_onset = None
        self.last_section = None
        self.last_period = None
        self.note_count = 0
        self.finished = False

    def _numeric_tokens(self,prefix):
        result={}
        for token,idx in self.vocab.items():
            if token.startswith(prefix+"_"):
                try:
                    value=int(token.split("_", 1)[1])
                    result[value]=idx
                except ValueError:
                    pass
        return result

    def _position(self,division,tatum):
        return Fraction(tatum-1, division)

    def _future(self,bar,beat,division,tatum):
        if self.last_onset is None:
            return True
        return (bar,beat,self._position(division,tatum)) > self.last_onset

    def _period_options(self, bar):
        if self.period_by_bar is not None:
            return (self.period_by_bar[bar],) if bar in self.period_by_bar else ()
        if self.beats_per_bar is not None:
            return (self.beats_per_bar,)
        # Without a source meter map, retain the first generated meter.
        if self.last_period is not None:
            return (self.last_period,)
        return tuple(period for period in self.periods if period >= 1)

    def _same_bar_has_future(self):
        if self.last_onset is None:
            return True
        _, last_beat, last_position = self.last_onset
        return (
            any(last_beat < beat <= self.last_period for beat in self.beats)
            or any(position > last_position for _, _, position in self.grids)
        )

    def _legal_bars(self, section):
        ids = []
        for bar, idx in sorted(self.bars.items()):
            if self.last_onset is None:
                if bar != self.start_bar:
                    continue
            else:
                last_bar = self.last_onset[0]
                if bar not in (last_bar, last_bar + 1):
                    continue
                if bar == last_bar:
                    if section != self.last_section or not self._same_bar_has_future():
                        continue
            if self.section_by_bar is not None:
                if self.section_by_bar.get(bar) != section:
                    continue
            if bar != (self.last_onset[0] if self.last_onset else None):
                if not any(
                    any(1 <= beat <= period for beat in self.beats)
                    for period in self._period_options(bar)
                ):
                    continue
            ids.append(idx)
        return ids

    def allowed_ids(self):
        if self.finished:
            return []
        field = self.fields[self.stage]
        if field == "SECTION":
            ids = [idx for idx in self.groups["SECTION"]
                   if self._legal_bars(self.tokens[idx])]
            if (self.eos_id is not None and self.last_onset is not None
                    and (not ids or (self.max_bar is not None
                                    and self.last_onset[0] >= self.max_bar))):
                ids.append(self.eos_id)
            return ids
        if field == "BAR":
            return self._legal_bars("SECTION_" + self.section)
        if field == "PERIOD":
            return [self.periods[period] for period in self._period_options(self.bar)
                    if period in self.periods and period >= 1
                    and (self.last_onset is None or self.bar != self.last_onset[0]
                         or period == self.last_period)]
        if field == "BEAT":
            ids = []
            for beat, idx in self.beats.items():
                if not 1 <= beat <= self.period:
                    continue
                if self.last_onset is not None and self.bar == self.last_onset[0]:
                    last_beat = self.last_onset[1]
                    if beat < last_beat:
                        continue
                    if beat == last_beat and not any(
                        position > self.last_onset[2]
                        for _, _, position in self.grids
                    ):
                        continue
                ids.append(idx)
            return ids
        if field == "DIVISION":
            return [idx for division, idx in self.divisions.items()
                    if division >= 1
                    and (self.max_division is None or division <= self.max_division)
                    and any(self._future(self.bar, self.beat, division, tatum)
                            for d, tatum, _ in self.grids if d == division)]
        if field == "TATUM":
            return [idx for tatum, idx in self.tatums.items()
                    if 1 <= tatum <= self.division
                    and self._future(self.bar, self.beat, self.division, tatum)]
        return self.groups[field]

    def advance(self,token_id):
        if token_id == self.eos_id:
            if self.stage != 0:
                raise ValueError("<EOS> cannot end an incomplete note")
            self.finished = True
            return
        field = self.fields[self.stage]
        token = self.tokens[token_id]
        if field == "SECTION":
            self.section = token.split("_",1)[1]
        if field in ("BAR", "PERIOD", "BEAT", "DIVISION", "TATUM"):
            setattr(self, field.lower(), int(token.split("_", 1)[1]))
        self.stage += 1
        if self.stage == len(self.fields):
            self.last_onset = (
                self.bar, self.beat, self._position(self.division, self.tatum)
            )
            self.last_period = self.period
            self.last_section = "SECTION_" + self.section
            self.note_count += 1
            self.stage = 0

def audit_melody_tokens(tokens):
    """Check note order and bar metadata independently of sampling state."""
    current = {}
    onsets = []
    invalid_grid = 0
    missing = 0
    bar_periods = {}
    bar_sections = {}
    period_changes_within_bar = 0
    section_changes_within_bar = 0
    skipped_bars = 0
    time_fields = ("BAR", "PERIOD", "BEAT", "DIVISION", "TATUM")
    previous_bar = None
    unfinished_event_tokens = 0
    pending_required_fields = set()
    for token in tokens:
        if token == "<EOS>":
            unfinished_event_tokens += len(current) + len(pending_required_fields)
            current = {}
            pending_required_fields.clear()
            continue
        if "_" not in token:
            continue
        field, value = token.split("_", 1)
        if field == "SECTION" and pending_required_fields:
            unfinished_event_tokens += len(pending_required_fields)
            pending_required_fields.clear()
        if field in time_fields or field == "SECTION":
            current[field] = value
        elif field == "PITCH":
            if any(field not in current for field in time_fields):
                missing += 1
                current = {}
                continue
            bar = int(current["BAR"])
            period = int(current["PERIOD"])
            beat = int(current["BEAT"])
            division = int(current["DIVISION"])
            tatum = int(current["TATUM"])
            if (period < 1 or not 1 <= beat <= period or division < 1
                    or not 1 <= tatum <= division):
                invalid_grid += 1
            if bar in bar_periods and bar_periods[bar] != period:
                period_changes_within_bar += 1
            bar_periods.setdefault(bar, period)
            if "SECTION" in current:
                if bar in bar_sections and bar_sections[bar] != current["SECTION"]:
                    section_changes_within_bar += 1
                bar_sections.setdefault(bar, current["SECTION"])
            if previous_bar is not None and bar > previous_bar + 1:
                skipped_bars += bar - previous_bar - 1
            previous_bar = bar
            position = Fraction(tatum - 1, division)
            onsets.append((bar, beat, position))
            current = {}
            pending_required_fields = {"DURATION", "VELOCITY"}
        elif field in ("DURATION", "VELOCITY"):
            pending_required_fields.discard(field)
    unfinished_event_tokens += len(current) + len(pending_required_fields)
    counts = Counter(onsets)
    return {
        "note_count": len(onsets),
        "missing_time_fields": missing,
        "position_mismatches": invalid_grid,
        "duplicate_onset_groups": sum(n > 1 for n in counts.values()),
        "extra_notes_at_same_onset": sum(n - 1 for n in counts.values()),
        "time_backward_steps": sum(b < a for a, b in zip(onsets, onsets[1:])),
        "skipped_bars": skipped_bars,
        "period_changes_within_bar": period_changes_within_bar,
        "section_changes_within_bar": section_changes_within_bar,
        "unfinished_event_tokens": unfinished_event_tokens,
    }


class JazzGenerationModel(nn.Module):
    def __init__(self, harmony_encoder, melody_decoder):
        super().__init__()
        self.harmony_encoder = harmony_encoder
        for p in self.harmony_encoder.parameters():
            p.requires_grad = False
        self.melody_decoder = melody_decoder

    def forward(self, harmony, melody_input):
        memory = self.harmony_encoder.encode(**harmony)
        return self.melody_decoder(melody_input, memory)

    @torch.no_grad()
    def encode_harmony(self, harmony):
        self.eval()
        return self.harmony_encoder.encode(**harmony)

    @torch.no_grad()
    def generate(
        self,
        harmony,
        prompt_ids,
        eos_id=None,
        max_length=2048,
        temperature=1.0,
        top_k=80,
        *,
        token_to_id=None,
        beats_per_bar=None,
        start_bar=None,
        max_bar=None,
        section_by_bar=None,
        period_by_bar=None,
        max_division=None,
        seed=None,
        return_info=False,
        chords = None,
        harmony_generator = None,
        dynamic_harmony = False
        ):


        if token_to_id is None:
            raise ValueError("token_to_id required")
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must be [1,length]")
        bos_id = token_to_id["<BOS>"]
        if prompt_ids[0,0].item()!=bos_id:
            raise ValueError( "Prompt must start with BOS" )
        constraint = SoloTokenConstraint(
            token_to_id,
            eos_id=eos_id,
            beats_per_bar=beats_per_bar,
            start_bar=start_bar,
            max_bar=max_bar,
            section_by_bar=section_by_bar,
            period_by_bar=period_by_bar,
            max_division=max_division,
        )
        self.eval()

        memory = self.harmony_encoder.encode(**harmony)
        note_chords = []
        generated = prompt_ids.to(memory.device)
        rng=None
        if seed is not None:
            rng=torch.Generator( device=memory.device)
            rng.manual_seed(seed)
        stop_reason="token_budget"
        while generated.size(1)<max_length:
        # -------------------------
        # prevent incomplete note
        # -------------------------
            if constraint.stage==0:
                remain=max_length-generated.size(1)
                if remain < len(constraint.fields)+1:
                    break
            allowed=constraint.allowed_ids()


            if not allowed:
                stop_reason = "no_future_position"
                if constraint.stage != 0:
                    raise RuntimeError(
                        f"No legal token at field {constraint.fields[constraint.stage]}; "
                        "the constraint entered an incomplete note"
                    )
                break



        # =========================
        # keep musical condition
        # =========================
            max_ctx=self.melody_decoder.max_seq_len
            if generated.size(1)>max_ctx:
                decoder_input=torch.cat([ generated[:,:3],generated[:,-(max_ctx-3):]],dim=1)
            else:
                decoder_input=generated
            logits=self.melody_decoder(decoder_input, memory)
            scores=logits[0,-1].clone()
            candidate_ids=torch.tensor(allowed, device=scores.device )
            scores=scores[candidate_ids]
            finite=torch.isfinite( scores)
            candidate_ids=candidate_ids[finite]
            scores=scores[finite]

            if scores.numel()==0:
                raise RuntimeError( "No legal token" )

        # =========================
        # sampling
        # =========================
            if temperature==0:
                next_id=candidate_ids[ scores.argmax()].item()

            else:
                scores=scores/temperature
                if top_k>0:
                    k=min(top_k,scores.numel())
                    scores,idx=torch.topk(scores,k)
                    candidate_ids=candidate_ids[idx]
                probs=torch.softmax(scores,dim=-1)
                choice=torch.multinomial(probs,1,generator=rng )
                next_id=candidate_ids[choice].item()
            next_token = constraint.tokens[next_id]
            constraint.advance(next_id)
            generated=torch.cat([generated, generated.new_tensor([[next_id]] )],dim=1)
# ==========================================
# Dynamic note-to-chord conditioning
#
# 一旦当前note的BAR已经确定，
# 就能知道这个note属于哪个chord。
#
# 之后 PERIOD / BEAT / DIVISION /
# TATUM / PITCH 等都会使用更新后的
# harmony memory。
# ==========================================
            if (dynamic_harmony 
                and chords is not None 
                and harmony_generator is not None
                and next_token.startswith("BAR_")
                ):
                current_bar = int(next_token.split("_",1)[1])
                currrent_chord = harmony_generator.chord_for_bar(chords,current_bar)
                note_chords.append(currrent_chord)
                note_harmony = harmony_generator.build_note_aligned(note_chords)
                note_harmony = {k:v.to(generated.device) for k, v in note_harmony.items()}
                memory = self.harmony_encoder.encode(**note_harmony)
            if constraint.finished:
                stop_reason="model_eos"
                break

    # =========================
    # force EOS safely
    # =========================
        forced_eos=False
        if eos_id is not None and not constraint.finished:
            if constraint.stage!=0:
                # Remove every field of the partial note, not just one field.
                generated = generated[:, :-constraint.stage]
                if dynamic_harmony and note_chords and constraint.stage >= 2:
                    note_chords.pop()
                constraint.stage = 0
            constraint.advance(eos_id)
            generated=torch.cat([ generated, generated.new_tensor( [[eos_id]] )],dim=1)
            forced_eos=True
        info={
            "stop_reason": stop_reason,
            "forced_eos": forced_eos,
            "note_count": constraint.note_count,
            "note_harmony_count":len(note_chords),
            "dynamic_harmony":dynamic_harmony,
            "token_count":generated.size(1),
            "temperature":temperature,
            "top_k":top_k,
            "seed":seed,
            "max_bar":max_bar
        }
        return ( generated, info) if return_info else generated

