"""Harmony-conditioned solo generation with constraints applied before sampling.

This uses the note format in your uploaded output:
BAR, BEAT, TATUM, POSITION, PITCH, DURATION, VELOCITY, [ARTIC], [MICRO].
ARTIC/MICRO are included if those token families exist in the vocabulary.
BEAT and TATUM are 1-based. TATUM values come from the vocabulary, not an
assumed four subdivisions per beat. No duration-unit conversion is assumed.

The neural layers and checkpoint parameter names are unchanged.
"""

import re
from collections import Counter

import torch
import torch.nn as nn


class SoloTokenConstraint:
    """Keep complete note onsets strictly increasing in (bar, beat, tatum).

    This intentionally emits melody note events only. Harmony is supplied to
    the encoder; harmony/header tokens are not emitted by this note grammar.
    It prevents equal onsets, not duration overlap between successive notes.
    """
    def __init__(self, token_to_id, eos_id=None, beats_per_bar=None,
                 start_bar=None, max_bar=None):
        self.vocab = dict(token_to_id)
        if (not self.vocab or any(type(i) is not int or i < 0
                                 for i in self.vocab.values())
                or len(set(self.vocab.values())) != len(self.vocab)):
            raise ValueError("token_to_id must contain unique nonnegative integer IDs.")
        self.tokens = {i: t for t, i in self.vocab.items()}
        if eos_id is not None and self.tokens.get(eos_id) != "<EOS>":
            raise ValueError("eos_id does not match <EOS> in this vocabulary.")
        if beats_per_bar is not None and (
                type(beats_per_bar) is not int or beats_per_bar < 1):
            raise ValueError("beats_per_bar must be a positive integer or None.")
        for name, value in (("start_bar", start_bar), ("max_bar", max_bar)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None.")
        self.eos_id = eos_id
        self.fields = ["BAR", "BEAT", "TATUM", "POSITION", "PITCH",
                       "DURATION", "VELOCITY"]
        self.groups = {}
        for field in self.fields + ["ARTIC", "MICRO"]:
            self.groups[field] = [i for t, i in self.vocab.items()
                                  if t.startswith(field + "_")]
        for field in self.fields:
            if not self.groups[field]:
                raise ValueError(f"Vocabulary has no {field}_ tokens.")
        for field in ("ARTIC", "MICRO"):
            if self.groups[field]:
                self.fields.append(field)

        self.bars = {}
        for token, idx in self.vocab.items():
            match = re.fullmatch(r"BAR_(\d+)", token)
            if match:
                bar = int(match.group(1))
                if (start_bar is None or bar >= start_bar) and (
                        max_bar is None or bar <= max_bar):
                    self.bars[bar] = idx
        if not self.bars or (start_bar is not None and start_bar not in self.bars):
            raise ValueError("No usable BAR tokens for start_bar/max_bar.")
        self.start_bar = start_bar

        self.pairs = []
        for token in self.vocab:
            match = re.fullmatch(r"POSITION_(\d+)-(\d+)", token)
            if not match:
                continue
            beat, tatum = map(int, match.groups())
            if (beat >= 1 and tatum >= 1
                    and (beats_per_bar is None or beat <= beats_per_bar)
                    and f"BEAT_{beat}" in self.vocab
                    and f"TATUM_{tatum}" in self.vocab):
                self.pairs.append((beat, tatum))
        self.pairs = sorted(set(self.pairs))
        if not self.pairs:
            raise ValueError("No matching BEAT/TATUM/POSITION tokens in vocabulary.")

        self.stage = 0
        self.bar = self.beat = self.tatum = None
        self.last_onset = None
        self.note_count = 0
        self.finished = False

    def _future_pairs(self, bar):
        return [pair for pair in self.pairs
                if self.last_onset is None or (bar, *pair) > self.last_onset]

    def allowed_ids(self):
        if self.finished:
            return []
        field = self.fields[self.stage]
        if field == "BAR":
            allowed = [idx for bar, idx in sorted(self.bars.items())
                       if self._future_pairs(bar)
                       and (self.last_onset is not None or self.start_bar is None
                            or bar == self.start_bar)]
            # EOS is legal only between complete notes, after at least one note.
            if self.note_count and self.eos_id is not None:
                allowed.append(self.eos_id)
            return allowed
        if field == "BEAT":
            return [self.vocab[f"BEAT_{beat}"] for beat in
                    sorted({b for b, _ in self._future_pairs(self.bar)})]
        if field == "TATUM":
            return [self.vocab[f"TATUM_{tatum}"] for beat, tatum in
                    self._future_pairs(self.bar) if beat == self.beat]
        if field == "POSITION":
            # This redundant field is determined by the selected beat/tatum.
            return [self.vocab[f"POSITION_{self.beat}-{self.tatum}"]]
        return self.groups[field]

    def advance(self, token_id):
        if token_id not in self.allowed_ids():
            raise ValueError(f"Illegal token {self.tokens.get(token_id, token_id)!r} "
                             f"at stage {self.fields[self.stage]}.")
        if token_id == self.eos_id:
            self.finished = True
            return
        field = self.fields[self.stage]
        if field in ("BAR", "BEAT", "TATUM"):
            setattr(self, field.lower(), int(self.tokens[token_id].split("_", 1)[1]))
        self.stage += 1
        if self.stage == len(self.fields):
            self.last_onset = (self.bar, self.beat, self.tatum)
            self.note_count += 1
            self.stage = 0


def audit_melody_tokens(tokens):
    """Read note onsets independently of the sampling state machine."""
    current = {}
    onsets = []
    positions = []
    inconsistent = missing = 0
    for token in tokens:
        if "_" not in token:
            continue
        field, value = token.split("_", 1)
        if field in ("BAR", "BEAT", "TATUM", "POSITION"):
            current[field] = value
        elif field == "PITCH":
            if any(k not in current for k in ("BAR", "BEAT", "TATUM", "POSITION")):
                missing += 1
                continue
            bar, beat, tatum = (int(current[k]) for k in ("BAR", "BEAT", "TATUM"))
            onsets.append((bar, beat, tatum))
            positions.append((bar, current["POSITION"]))
            inconsistent += current["POSITION"] != f"{beat}-{tatum}"
    counts = Counter(onsets)
    return {
        "note_count": len(onsets),
        "missing_time_fields": missing,
        "position_mismatches": inconsistent,
        "duplicate_onset_groups": sum(n > 1 for n in counts.values()),
        "extra_notes_at_same_onset": sum(n - 1 for n in counts.values()),
        "duplicate_position_groups": sum(n > 1 for n in Counter(positions).values()),
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
    def generate(self, harmony, bos_id, eos_id=None, max_length=512,
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
        if token_to_id is None or token_to_id.get("<BOS>") != bos_id:
            raise ValueError("Pass the checkpoint's token_to_id; bos_id must match <BOS>.")
        if not (temperature >= 0 and temperature < float("inf")):
            raise ValueError("temperature must be finite and >= 0.")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be an integer >= 0.")
        constraint = SoloTokenConstraint(token_to_id, eos_id, beats_per_bar,
                                         start_bar, max_bar)
        reserve_eos = int(eos_id is not None)
        if type(max_length) is not int or max_length < 1 + len(constraint.fields) + reserve_eos:
            raise ValueError("max_length must fit BOS, one complete note, and EOS.")

        self.eval()
        memory = self.harmony_encoder.encode(**harmony)
        generated = torch.tensor([[bos_id]], dtype=torch.long, device=memory.device)
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
