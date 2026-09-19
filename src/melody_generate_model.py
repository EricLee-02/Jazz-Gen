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
    

    def __init__(self,token_to_id,eos_id=None,beats_per_bar=None,start_bar=None,max_bar=None):
        self.vocab = dict(token_to_id)
        self.tokens = {i: t for t, i in self.vocab.items()}
        if (not self.vocab or any(type(i) is not int or i < 0 for i in self.vocab.values())or len(set(self.vocab.values())) != len(self.vocab)):
            raise ValueError("token_to_id must contain unique nonnegative integer IDs.")
        if eos_id is not None and self.tokens.get(eos_id) != "<EOS>":
            raise ValueError( "eos_id does not match <EOS> in this vocabulary.")
        if beats_per_bar is not None and (type(beats_per_bar) is not int or beats_per_bar < 1):
            raise ValueError("beats_per_bar must be a positive integer or None.")
        for name, value in (("start_bar", start_bar), ("max_bar", max_bar)):
            if value is not None and ( type(value) is not int or value < 0):
                raise ValueError( f"{name} must be a nonnegative integer or None.")

        self.eos_id = eos_id
        self.beats_per_bar = beats_per_bar
        self.start_bar = start_bar
        self.max_bar = max_bar

        # ==========================================
        # New note grammar
        # ==========================================

        self.fields = [
            "BAR",
            "PERIOD",
            "BEAT",
            "DIVISION",
            "TATUM",
            "PITCH",
            "DURATION",
            "VELOCITY"
        ]

        self.groups = {}

        for field in self.fields + ["ARTIC", "MICRO"]:

            self.groups[field] = [
                idx
                for token, idx in self.vocab.items()
                if token.startswith(field + "_")
            ]

        # Required token families
        for field in self.fields:

            if not self.groups[field]:
                raise ValueError(
                    f"Vocabulary has no {field}_ tokens."
                )

        # Optional note fields
        for field in ("ARTIC", "MICRO"):

            if self.groups[field]:
                self.fields.append(field)

        # ==========================================
        # Numeric token maps
        # ==========================================

        self.bars = self._numeric_tokens("BAR")
        self.periods = self._numeric_tokens("PERIOD")
        self.beats = self._numeric_tokens("BEAT")
        self.divisions = self._numeric_tokens("DIVISION")
        self.tatums = self._numeric_tokens("TATUM")

        # Filter BAR range
        self.bars = {
            value: idx
            for value, idx in self.bars.items()
            if (
                (start_bar is None or value >= start_bar)
                and
                (max_bar is None or value <= max_bar)
            )
        }

        if not self.bars:
            raise ValueError(
                "No usable BAR tokens for start_bar/max_bar."
            )

        if (
            start_bar is not None
            and start_bar not in self.bars
        ):
            raise ValueError(
                f"BAR_{start_bar} is not in the vocabulary."
            )

        # ==========================================
        # Generation state
        # ==========================================

        self.stage = 0
        self.bar = None
        self.period = None
        self.beat = None
        self.division = None
        self.tatum = None
        # (bar, beat, Fraction)
        self.last_onset = None

        self.note_count = 0
        self.finished = False


    def _numeric_tokens(self, prefix):

        result = {}

        pattern = re.compile(
            rf"{prefix}_(\d+)"
        )

        for token, idx in self.vocab.items():

            match = pattern.fullmatch(token)

            if match:

                result[int(match.group(1))] = idx

        return result


    def _position_fraction(self,division,tatum):
        # TATUM is 1-based
        return Fraction(tatum - 1,division)


    def _is_future(self, bar, beat, division,tatum):

        if self.last_onset is None:
            return True

        onset = (bar, beat,self._position_fraction(division,tatum) )

        return onset > self.last_onset


    def allowed_ids(self):

        if self.finished:
            return []
        field = self.fields[self.stage]
        # ==========================================
        # BAR
        # ==========================================

        if field == "BAR":

            allowed = []

            for bar, idx in sorted(self.bars.items() ):
                if self.last_onset is None:

                    if ( self.start_bar is None or bar == self.start_bar):
                        allowed.append(idx)
                else:
                    # Same bar or later bar
                    if bar >= self.last_onset[0]:
                        allowed.append(idx)

            # EOS only between complete notes
            if ( self.note_count > 0 and self.eos_id is not None):
                allowed.append(self.eos_id )
            return allowed

        # ==========================================
        # PERIOD
        # ==========================================

        if field == "PERIOD":
            return [ idx for period, idx in sorted(self.periods.items()) if period >= 1 ]


        # ==========================================
        # BEAT
        # ==========================================

        if field == "BEAT":

            if self.beats_per_bar is not None:

                beat_limit = self.beats_per_bar

            else:

                # WJazzD period normally represents
                # the number of beats in the metrical period
                beat_limit = self.period

            allowed = []

            for beat, idx in sorted(
                self.beats.items()
            ):

                if beat < 1:
                    continue

                if (
                    beat_limit is not None
                    and beat > beat_limit
                ):
                    continue

                if self.last_onset is not None:

                    last_bar, last_beat, _ = (
                        self.last_onset
                    )

                    if self.bar == last_bar:

                        if beat < last_beat:
                            continue

                allowed.append(idx)

            return allowed


        # ==========================================
        # DIVISION
        # ==========================================

        if field == "DIVISION":

            allowed = []

            for division, idx in sorted(
                self.divisions.items()
            ):

                if division < 1:
                    continue

                # At least one valid tatum must
                # exist for this division
                valid_tatum_exists = False

                for tatum in self.tatums:

                    if not 1 <= tatum <= division:
                        continue

                    if self._is_future(
                        self.bar,
                        self.beat,
                        division,
                        tatum
                    ):

                        valid_tatum_exists = True
                        break

                if valid_tatum_exists:

                    allowed.append(idx)

            return allowed


        # ==========================================
        # TATUM
        # ==========================================

        if field == "TATUM":
            allowed = []
            for tatum, idx in sorted(self.tatums.items()):

                # Important relationship:
                # TATUM cannot exceed DIVISION
                if not 1 <= tatum <= self.division:
                    continue

                if self._is_future(
                    self.bar,
                    self.beat,
                    self.division,
                    tatum
                ):

                    allowed.append(idx)

            return allowed


        # ==========================================
        # Other note fields
        # ==========================================

        return self.groups[field]


    def advance(self, token_id):

        allowed = self.allowed_ids()

        if token_id not in allowed:

            raise ValueError(
                f"Illegal token "
                f"{self.tokens.get(token_id, token_id)!r} "
                f"at stage {self.fields[self.stage]}."
            )

        if token_id == self.eos_id:

            self.finished = True
            return


        field = self.fields[self.stage]


        if field in (
            "BAR",
            "PERIOD",
            "BEAT",
            "DIVISION",
            "TATUM"
        ):

            value = int(
                self.tokens[token_id].split(
                    "_",
                    1
                )[1]
            )

            setattr(
                self,
                field.lower(),
                value
            )


        self.stage += 1


        # A complete note has been produced
        if self.stage == len(self.fields):
            onset = (self.bar,self.beat, self._position_fraction(self.division,self.tatum))

            if (self.last_onset is not None and onset <= self.last_onset):
                raise RuntimeError(
                    f"Generated onset is not increasing: "
                    f"{onset} <= {self.last_onset}"
                )

            self.last_onset = onset
            self.note_count += 1
            self.stage = 0

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
    def generate(self, harmony,prompt_ids, eos_id=None, max_length=512,
                 temperature=0.8, top_k=20, *, token_to_id=None,
                 beats_per_bar=None, start_bar=None, max_bar=None,
                 seed=None, return_info=False):
        """Generate one solo; return shape [1, length], as in the original API.

        token_to_id is now required for time/format constraints. max_length
        includes BOS/EOS and must fit the decoder's positional capacity.
        temperature=0 selects argmax; top_k=0 samples all LEGAL candidates.
        At a token limit, stop at a complete note and append EOS when supplied.
        This can end before the harmony sequence ends; see stop_reason.
        """
        # if token_to_id is None or token_to_id.get("<BOS>") != bos_id:
        #     raise ValueError("Pass the checkpoint's token_to_id; bos_id must match <BOS>.")
        if token_to_id is None:
            raise ValueError("Pass checkpoint token_to_id")
        
        if (prompt_ids.ndim != 2 or prompt_ids.size(0) != 1):
            raise ValueError("prompt_ids must have shape [1, prompt_length].")
        
        bos_id = token_to_id["<BOS>"]

        if prompt_ids[0,0].item() != bos_id :
            raise ValueError("prompt_ids must start with <BOS>.")

        if not (temperature >= 0 and temperature < float("inf")):
            raise ValueError("temperature must be finite and >= 0.")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be an integer >= 0.")
        constraint = SoloTokenConstraint(token_to_id, eos_id, beats_per_bar,
                                         start_bar, max_bar)
        reserve_eos = int(eos_id is not None)
        if type(max_length) is not int or max_length < prompt_ids.size(1)+len(constraint.fields)+reserve_eos:
            raise ValueError("max_length must fit BOS, one complete note, and EOS.")

        self.eval()
        memory = self.harmony_encoder.encode(**harmony)
        #generated = torch.tensor([[bos_id]], dtype=torch.long, device=memory.device)
        generated = prompt_ids.to(memory.device)
        rng = None
        if seed is not None:
            rng = torch.Generator(device=memory.device)
            rng.manual_seed(seed)
        stop_reason = "token_budget"

        while generated.size(1) < max_length:
            # Do not begin a note that would be cut halfway by max_length.
            if constraint.stage == 0 and (
                    generated.size(1) + len(constraint.fields) + reserve_eos > max_length):
                break
            allowed = constraint.allowed_ids()
            if not allowed:
                stop_reason = "no_future_position"
                break
            logits = self.melody_decoder(generated, memory)
            if logits.ndim != 3 or logits.size(0) != 1:
                raise ValueError("generate expects decoder logits shaped [1, length, vocab].")
            # Equivalent to setting every illegal token's logit to -inf.
            # Apply this BEFORE temperature/top-k, never after sampling.
            candidate_ids = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            scores = logits[0, -1, candidate_ids].float()
            if torch.isnan(scores).any() or torch.isposinf(scores).any():
                raise RuntimeError("Decoder produced NaN/+inf on legal candidates.")
            finite = torch.isfinite(scores)
            candidate_ids, scores = candidate_ids[finite], scores[finite]
            if scores.numel() == 0:
                raise RuntimeError("No finite legal candidate. Constraints were not bypassed.")
            if temperature == 0:
                next_id = candidate_ids[scores.argmax()].item()
            else:
                scores = scores / temperature
                if top_k:
                    scores, order = torch.topk(scores, min(top_k, scores.numel()))
                    candidate_ids = candidate_ids[order]
                probs = torch.softmax(scores, dim=-1)
                choice = torch.multinomial(probs, 1, generator=rng)
                next_id = candidate_ids[choice].item()
            constraint.advance(next_id)
            generated = torch.cat((generated, generated.new_tensor([[next_id]])), dim=1)
            if constraint.finished:
                stop_reason = "no_future_position" if allowed == [eos_id] else "model_eos"
                break

        forced_eos = False
        if eos_id is not None and not constraint.finished:
            # Loop budgeting guarantees a note boundary and one spare EOS slot.
            if constraint.stage != 0 or generated.size(1) >= max_length:
                raise RuntimeError("Generation stopped inside a note event.")
            constraint.advance(eos_id)
            generated = torch.cat((generated, generated.new_tensor([[eos_id]])), dim=1)
            forced_eos = True

        info = {
            "stop_reason": stop_reason, "forced_eos": forced_eos,
            "note_count": constraint.note_count, "token_count": generated.size(1),
            "temperature": temperature, "top_k": top_k, "seed": seed,
            "beats_per_bar": beats_per_bar, "start_bar": start_bar,
            "max_bar": max_bar, "note_fields": constraint.fields,
        }
        return (generated, info) if return_info else generated
