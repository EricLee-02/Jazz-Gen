


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
from .config import GRID_PER_BEAT

"""Keep complete note onsets strictly increasing in (bar, beat, tatum).

This intentionally emits melody note events only. Harmony is supplied to
the encoder; harmony/header tokens are not emitted by this note grammar.
It prevents equal onsets, not duration overlap between successive notes.
"""

class SoloTokenConstraint:

    def __init__(
        self,
        token_to_id,
        eos_id=None,
        beats_per_bar=None,
        start_bar=None,
        max_bar=None
    ):
        self.min_onset_gap = Fraction(8,24)
        self.max_onset_gap = Fraction(2,1)
        self.max_division = 3

        self.vocab = dict(token_to_id)
        self.tokens = {i:t for t,i in self.vocab.items()}
        self.eos_id=eos_id
        self.beats_per_bar=beats_per_bar
        self.start_bar=start_bar
        self.max_bar=max_bar
        self.fields=[
            "SECTION",
            "BAR",
            "PERIOD",
            "BEAT",
            "DIVISION",
            "TATUM",
            "PITCH",
            "DURATION",
            "VELOCITY"
        ]
        self.groups={}
        for field in self.fields + ["ARTIC","MICRO"]:
            self.groups[field]=[idx for token,idx in self.vocab.items() if token.startswith( field+"_")]
        for field in self.fields:
            if not self.groups[field]:
                raise ValueError( f"Missing {field} vocabulary")

        if self.groups["ARTIC"]:
            self.fields.append( "ARTIC")
        if self.groups["MICRO"]:
            self.fields.append( "MICRO")
        self.bars=self._numeric_tokens("BAR")
        self.periods=self._numeric_tokens("PERIOD")
        self.beats=self._numeric_tokens("BEAT")
        self.divisions=self._numeric_tokens("DIVISION")
        self.tatums=self._numeric_tokens("TATUM")


        if max_bar is not None:
            self.bars={ k:v for k,v in self.bars.items() if k<=max_bar}
        self.stage=0
        self.section = None
        self.bar=None
        self.period=None
        self.beat=None
        self.division=None
        self.tatum=None
        self.last_onset=None
        self.note_count=0
        self.finished=False

    def _absolute_position(self,bar,beat,division,tatum):
        return (Fraction(bar * self.beats_per_bar+beat-1,1)+self._position(division=division,tatum=tatum))
    
    def _last_absolute_position(self):
        if self.last_onset is None:
            return None
        bar,beat,position = self.last_onset
        return (Fraction(bar * self.beats_per_bar + beat-1,1)+position)




    def _numeric_tokens(self,prefix):
        result={}
        for token,idx in self.vocab.items():
            if token.startswith(prefix+"_"):
                try:
                    value=int( token.split("_")[1])
                    result[value]=idx
                except:
                    pass
        return result
    





    def _position(self,division,tatum):
        position = canonical_position(division,tatum)

        return position



    def _future(self,bar,beat,division,tatum):
        #First note
        if self.last_onset is None:
            if beat < 1:
                return False
            if (self.beats_per_bar is not None
                and beat > self.beats_per_bar):
                return False
            if division < 1:
                return False
            if tatum < 1:
                return False
            if tatum > division:
                return False
            return True
        current = self._absolute_position(bar,beat,division,tatum)
        previous = self._last_absolute_position()
        gap = current - previous

        if gap <= 0 :
            return False
        if gap <self.min_onset_gap:
            return False
        if gap > self.max_onset_gap:
            return False
        return True
        # return (bar,beat,self._position(division,tatum)) > self.last_onset



    def allowed_ids(self):
        if self.finished:
            return []
        field=self.fields[self.stage]
        if field == "SECTION":
            ids = list(self.groups["SECTION"])
            if (
                self.eos_id is not None 
                and self.last_onset is not  None 
                and self.max_bar is not None 
                and self.last_onset[0]>= self.max_bar):
                ids.append(self.eos_id)
            return ids
        
        if field == "BAR":
            if self.last_onset is None:
                start = self.start_bar 
                if start not in self.bars:
                    raise ValueError(f"BAR_{start} not found in vocabulary")
                return [self.bars[start]]
            last_bar = self.last_onset[0]
            ids = []
            if self._bar_has_feature(last_bar):
                if last_bar in self.bars:
                    ids.append(self.bars[last_bar])
            next_bar = last_bar +1
            if (next_bar in self.bars and (self.max_bar is None or next_bar <= self.max_bar) and self._bar_has_feature(next_bar)):
                ids.append(self.bars[next_bar])

            return ids
        # =========================
        # PERIOD
        # =========================
        if field=="PERIOD":
            if self.beats_per_bar is not None:
                if self.beats_per_bar not in self.periods:
                    raise ValueError(f"PERIOD_{self.beats_per_bar} not found in vocabulary")
                return [self.periods[self.beats_per_bar]]
            
            if (self.last_onset is not None 
                and self.bar == self.last_onset[0]
                and self.period in self.periods):
                return [self.periods[self.period]]
            
            return list(self.groups["PERIOD"])
        # =========================
        # BEAT
        # =========================
        if field=="BEAT":
            ids=[]
            for beat,idx in self.beats.items():
                if beat<1:
                    continue
                if (self.period is not None
                    and beat > self.period):
                    continue
                if self._beat_has_feature(self.bar,beat):
                    ids.append(idx)
            return ids
        # =========================
        # DIVISION
        # =========================
        if field=="DIVISION":
            ids = []
            for division, idx in self.divisions.items():
                if division < 1:
                    continue
                if division > self.max_division:
                    continue
                has_future = False
                for tatum in self.tatums.keys():
                    if tatum < 1:
                        continue
                    if tatum > division:
                        continue
                    if self._future(self.bar,self.beat,division,tatum):
                        has_future=True
                        break
                if has_future:
                    ids.append(idx)
            return ids
        # =========================
        # TATUM
        # =========================
        if field=="TATUM":
            ids=[]
            for tatum,idx in self.tatums.items():
                if tatum < 1:
                    continue
                if tatum > self.division:
                    continue
                if self._future(self.bar,self.beat,self.division,tatum):
                    ids.append(idx)
            return ids
        # =========================
        # Melody tokens
        # =========================

        # Fully open:
        # let Transformer decide
        return self.groups[field]
    
    def _beat_has_feature(self,bar,beat):
        if beat < 1:
           return False
        if beat > self.beats_per_bar:
           return False
        for division in self.divisions.keys():
            if division < 1:
                continue
            if division > self.max_division:
                continue
            for tatum in self.tatums.keys():
                if tatum < 1:
                   continue
                if tatum > division:
                   continue
                if self._future(bar=bar,beat=beat,division=division,tatum=tatum):
                   return True
        return False
    
    def _bar_has_feature(self,bar):
        if (self.max_bar is not None
            and bar > self.max_bar):
            return False
        for beat in self.beats.keys():
            if beat < 1:
                continue
            if beat > self.beats_per_bar:
                continue
            if self._beat_has_feature(bar,beat):
                return True
        return False

    def _same_bar_has_feature(self):
        if self.last_onset is None:
            return True
        last_bar = self.last_onset[0]
        for beat in self.beats.keys():
            if beat < 1:
                continue
            if beat>self.beats_per_bar:
                continue
            if self._beat_has_feature(last_bar,beat):
                return True
           
        return False




    def advance(self,token_id):
        if token_id==self.eos_id:
            self.finished=True
            return
        field=self.fields[self.stage]
        token=self.tokens[token_id]
        if field == "SECTION":
            self.section = token.split("_",1)[1]
        if field in [
            "BAR",
            "PERIOD",
            "BEAT",
            "DIVISION",
            "TATUM"
            ]:
            value=int(token.split("_")[1])
            setattr( self,field.lower(), value)
        self.stage+=1
        if self.stage==len(self.fields):
            onset=(self.bar,self.beat,self._position(self.division,self.tatum))
            self.last_onset=onset
            self.note_count+=1
            self.stage=0

def apply_swing_position(division,tatum,position,swing_ratio=0.625):
    if division == 2 and tatum == 2:
        return Fraction(round(swing_ratio*24),24)
    return position

def canonical_position(division,tatum):
        division = int(division)
        tatum = int(tatum)
        if division < 1:
            raise ValueError(f"Invalid division: {division}")
        if tatum < 1:
            raise ValueError(f"Invalid tatum: {tatum}")
        if tatum > division:
            raise ValueError(f"TATUM_{tatum} > DIVISION_{division}")
    # ----------------------------------------
    # Original WJazzD position
    # ---------------------------------------
        raw_position = Fraction(tatum - 1,division)
    # ----------------------------------------
    # Map onto 24-grid
    # ----------------------------------------
        grid_index = round(raw_position*GRID_PER_BEAT)
        grid_index = int(grid_index)
    # Never allow position == 1 beat.
    #
    # The next beat itself should be represented
    # by BEAT_n+1 / TATUM_1.
    # ----------------------------------------
        grid_index = max(0,min(GRID_PER_BEAT - 1,grid_index))
        return Fraction(grid_index,GRID_PER_BEAT)


def audit_melody_tokens(tokens):
    """Read note onsets independently of the sampling state machine."""
    current = {}
    onsets = []
    invalid_grid = 0 
    missing = 0
    time_fileds = (
        "BAR",
        "PERIOD",
        "BEAT",
        "DIVISION",
        "TATUM"
    )
    for token in tokens:
        if "_" not in token:
            continue
        field, value = token.split("_", 1)
        if field in time_fileds:
            current[field] = value
        elif field == "PITCH":
            if any(field  not in current for field in time_fileds):
                missing += 1
                current = {}
                continue
            bar = int(current["BAR"])
            period = int(current["PERIOD"])
            beat = int(current["BEAT"])
            division = int(current["DIVISION"])
            tatum = int(current["TATUM"])
            if(period < 1 or beat < 1 or beat > period or division < 1 or tatum < 1 or tatum > division ):
                invalid_grid += 1
            position = canonical_position(division=division,tatum=tatum)
            onsets.append((bar, beat, position))
            current = {}
    counts = Counter(onsets)
    return {
        "note_count": len(onsets),
        "missing_time_fields": missing,
        "position_mismatches": invalid_grid,
        "duplicate_onset_groups": sum(n > 1 for n in counts.values()),
        "extra_notes_at_same_onset": sum(n - 1 for n in counts.values()),
        "time_backward_steps": sum(b < a for a, b in zip(onsets, onsets[1:])),
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
        seed=None,
        return_info=False,
        chords = None,
        harmony_generator = None,
        dynamic_harmony = False,
        duration_temperature = 1.15,
        duration_bias = None
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
        eos_id,
        beats_per_bar,
        start_bar,
        max_bar
        )
        self.eval()

        memory = self.harmony_encoder.encode(**harmony)
        note_chords = []
        generated = prompt_ids.to(memory.device)

        # Track duration choices so every generation report shows whether
        # the decoder is collapsing toward one duration token.
        duration_counter = Counter()

        # First-pass correction based on the observed training/generation
        # mismatch.  This is intentionally applied ONLY while sampling the
        # DURATION field, so pitch/rhythm/section sampling is unchanged.
        if duration_bias is None:
            duration_bias = {
                "DURATION_2": 0.10,
                "DURATION_4": 0.20,
                "DURATION_8": 0.70,
                "DURATION_16": -0.30,
                "DURATION_32": -0.60,
                "DURATION_LONG": -0.20,
            }

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
                    stop_reason = "token_budget"
                    break
            # Save the field BEFORE sampling/advance(), because advance()
            # immediately moves the grammar to the next field.
            current_field = constraint.fields[constraint.stage]
            allowed=constraint.allowed_ids()


            if not allowed:
                stop_reason="no_future_position"
                #debug
                print("=" * 60)
                print("NO FUTURE POSITION")
                print("field:",constraint.fields[constraint.stage ])
                print("stage:", constraint.stage)
                print("note_count:",constraint.note_count)
                print("bar:",constraint.bar)
                print( "period:",constraint.period)
                print( "beat:",constraint.beat)
                print( "division:", constraint.division)
                print( "tatum:",constraint.tatum)
                print("last_onset:", constraint.last_onset)
                print("=" * 60)
                stop_reason = "no_future_position"
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
            candidate_ids=torch.tensor(allowed,dtype=torch.long, device=scores.device )
            scores=scores[candidate_ids]
            finite=torch.isfinite( scores)
            candidate_ids=candidate_ids[finite]
            scores=scores[finite]

            if scores.numel()==0:
                raise RuntimeError( "No legal token" )

        # ==========================================================
        # DURATION-specific logit correction
        # ==========================================================
        # Training data: D16=51.42%, D8=17.15%
        # Recent generation: D16~72.8%, D8~7.2%
        # The correction is intentionally field-specific and does not alter
        # BAR/BEAT/DIVISION/TATUM/PITCH/VELOCITY probabilities.
            if current_field == "DURATION":
                for i, token_id in enumerate(candidate_ids.tolist()):
                    token = constraint.tokens[token_id]
                    scores[i] += duration_bias.get(token, 0.0)

        # =========================
        # field-specific temperature
        # =========================
        # Keep temperature=0 globally greedy. Otherwise, DURATION receives
        # a slightly higher temperature to reduce mode concentration.
            if temperature == 0:
                current_temperature = 0.0
            elif current_field == "DURATION":
                current_temperature = duration_temperature
            else:
                current_temperature = temperature

        # =========================
        # sampling
        # =========================
            if current_temperature==0:
                next_id=candidate_ids[scores.argmax()].item()

            else:
                scores=scores/current_temperature
                if top_k>0:
                    k=min(top_k,scores.numel())
                    scores,idx=torch.topk(scores,k)
                    candidate_ids=candidate_ids[idx]
                probs=torch.softmax(scores,dim=-1)
                choice=torch.multinomial(probs,1,generator=rng )
                next_id=candidate_ids[choice].item()
            next_token = constraint.tokens[next_id]

            if next_token.startswith("DURATION_"):
                duration_counter[next_token] += 1
            constraint.advance(next_id)
            generated=torch.cat([generated, generated.new_tensor([[next_id]] )],dim=1)
            # ==========================================
            # Dynamic NOTE-TO-CHORD conditioning
            #
            # Grammar:
            # SECTION -> BAR -> PERIOD -> BEAT -> DIVISION -> TATUM
            #         -> PITCH -> DURATION -> VELOCITY
            #
            # Wait until TATUM has been generated. At that point the exact
            # symbolic onset (bar/beat/division/tatum) is known, so a bar such
            # as "DMin7 G7" can resolve to the correct active chord BEFORE
            # PITCH is sampled.
            # ==========================================
            if (
                dynamic_harmony
                and chords is not None
                and harmony_generator is not None
                and current_field == "TATUM"
            ):
                current_meter = (
                    constraint.period
                    or beats_per_bar
                    or 4
                )

                current_chord = harmony_generator.chord_for_position(
                    bars=chords,
                    bar=constraint.bar,
                    beat=constraint.beat,
                    division=constraint.division,
                    tatum=constraint.tatum,
                    beats_per_bar=current_meter,
                )

                note_chords.append(current_chord)

                note_harmony = harmony_generator.build_note_aligned(
                    note_chords
                )
                note_harmony = {
                    k: v.to(generated.device)
                    for k, v in note_harmony.items()
                }
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
                incomplete_stage = constraint.stage
                remove_count = incomplete_stage
                if generated.size(1) >= remove_count:
                    generated = generated[:,:-remove_count]
                # ----------------------------------------------------
                # Dynamic harmony synchronization
                #
                # Harmony is now appended immediately AFTER TATUM is
                # generated.  At that moment constraint.stage points to
                # PITCH.  If token-budget cleanup discards such an incomplete
                # note, its note-aligned harmony event must also be removed.
                # ----------------------------------------------------
                harmony_appended_stage = constraint.fields.index("PITCH")
                if (
                    dynamic_harmony
                    and note_chords
                    and incomplete_stage >= harmony_appended_stage
                ):
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
            "note_chord_distribution":dict(Counter(note_chords)),
            "last_note_chord":(note_chords[-1] if note_chords else None),
            "dynamic_harmony":dynamic_harmony,
            "token_count":generated.size(1),
            "temperature":temperature,
            "duration_temperature":duration_temperature,
            "duration_bias":dict(duration_bias),
            "duration_distribution":dict(duration_counter),
            "duration_distribution_pct":({
                token: round(count / sum(duration_counter.values()) * 100.0, 2)
                for token, count in duration_counter.items()
            } if duration_counter else {}),
            "top_k":top_k,
            "seed":seed,
            "max_bar":max_bar
        }
        return ( generated, info) if return_info else generated
