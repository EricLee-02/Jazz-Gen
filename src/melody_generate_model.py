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

        return Fraction(tatum-1, division)



    def _future(self,bar,beat,division,tatum):
        if self.last_onset is None:
            return True
        return (bar,beat,self._position(division,tatum)) > self.last_onset



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
        # =========================
        # PERIOD
        # =========================
        if field=="PERIOD":
            return list(self.groups["PERIOD"])
        # =========================
        # BEAT
        # =========================
        if field=="BEAT":
            ids=[]
            for beat,idx in self.beats.items():
                if beat<1:
                    continue
                if self.period:
                    if beat>self.period:
                        continue
                ids.append(idx)
            return ids
        # =========================
        # DIVISION
        # =========================
        if field=="DIVISION":
            return list( self.groups["DIVISION"])
        # =========================
        # TATUM
        # =========================
        if field=="TATUM":
            ids=[]
            for tatum,idx in self.tatums.items():
                if tatum<=self.division:
                    if self._future(self.bar,self.beat,self.division,tatum):
                        ids.append(idx)
            return ids
        # =========================
        # Melody tokens
        # =========================

        # Fully open:
        # let Transformer decide
        return self.groups[field]



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
            position = Fraction(tatum-1, division)
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
        eos_id,
        beats_per_bar,
        start_bar,
        max_bar
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
                stop_reason="no_future_position"
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
            # discard incomplete event
                while (constraint.stage!=0 and generated.size(1)>0):
                    generated=generated[:,:-1]
                    break
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
