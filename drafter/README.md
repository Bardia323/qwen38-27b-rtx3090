# drafter/ — self-distillation data, calibrated int4 requant, and MTP fine-tuning

Tooling used to build the single-user "fast" variant of the model
(`models/Qwen3.8-27B-W4A16-AutoRound-fast`, prebuilt on the Hub as
`syvai/qwen3.8-27b-3090-fast-variant`; `fetch_fast_variant.py` assembles it). It also
contains a complete MTP-head fine-tuning pipeline that, honestly, did **not** move the
needle — kept because the negative result is informative and the same data feeds the
things that did work.

Everything here runs on the 3090 in the serving venv; ~6 h of GPU time end to end.

## What actually mattered (in order)

1. **The draft-head vocabulary.** The MTP drafter scores a 40,960-row slice of `lm_head`
   (`build_draft_vocab.py`); a token outside that slice can never be drafted, so every
   such token is a guaranteed rejection *and* truncates the chain. The originally shipped id
   list (counted over Danish web text, Wikipedia, Python, and 8.8M tokens of older outputs)
   covered 92.1% of what the model actually generates — 83% on code. A list counted over
   5.4M tokens of the model's own outputs (this pipeline, `gen_data.py`) covers 97.5%
   (96% on code). Nothing else changed: 98.0 → 108.6 tok/s greedy, 90.0 → 107.4 at
   default sampling. Coverage barely improves past 40k rows (49k: 98.2%; the model only
   ever emits ~54k distinct tokens), and 49k measured no faster.
2. **GPTQ-calibrated int4 for lm_head and the MTP module.** Round-to-nearest int4 costs
   +1.5% perplexity on lm_head (KL to the bf16 head 0.0068) and ~2% acceptance on the
   MTP module. GPTQ with a Hessian from 300k captured hidden states (`gptq_lm_head.py`)
   halves the lm_head KL (0.0029, +0.6% PPL, GSM8K 96.5% unchanged) and the MTP
   Hessians from `train_mtp.py --dump-hessians` keep acceptance intact
   (`requant_mtp_gptq.py`). Together: −1.8 ms per decode step (108.6 → 118.8 greedy).
3. **Fine-tuning the MTP head: no.** Distilling the target's own distribution into the
   drafter (KL over the draft vocab, unrolled depth-2 chains, 7M tokens, one epoch)
   halves the KL and looks great on the naive metric — until you score only response
   tokens against the *actual* next token, where top-1 agreement is unchanged
   (0.685 → 0.685) and vLLM's acceptance moves within noise. Qwen's head is already at the
   ceiling of a single-layer chain drafter for greedy top-1 on this data; the KL gains
   were on prompt tokens and on positions whose true token is outside the draft vocab.
   `train_mtp.py --eval-only` with `--depths 4` prints the greedy chain simulation that
   matches vLLM (2.5 vs 2.6 tok/step for the original head).

## Pipeline

```bash
V=venv/bin/python
$V drafter/collect_prompts.py                 # 6.8k prompts: UltraChat, Magicoder, syvai/da-instruction,
                                              #   syvai/reasoning-v1, skolegpt-instruct, GSM8K; 45% thinking on
VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp $V drafter/gen_data.py   # 2.2 h, 5.4M output tokens
$V drafter/capture.py                         # 1.7 h: hidden states of every token (74 GB memmap), in-process
                                              #   vLLM hook on GPUModelRunner._model_forward
# draft vocab from the model's own outputs -> draft_vocab_ids.json (the shipped list)
$V drafter/train_mtp.py --out drafter/runs/e --eval-only 1 --draft-ids draft_vocab_ids.json \
     --max-seqs 400 --val-frac 0.4 --depths 2 --dump-hessians drafter/runs/e/mtp_hessians.pt
$V drafter/gptq_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound models/tmp-lm4 --bits 4 --calib-rows 300000
$V build_draft_vocab.py models/tmp-lm4 --ids draft_vocab_ids.json      # int4 draft head from the int4 lm_head
$V drafter/requant_mtp_gptq.py models/tmp-lm4 models/Qwen3.8-27B-W4A16-AutoRound-fast drafter/runs/e/mtp_hessians.pt --bits 4
```

Optional fine-tune (for the record): `train_mtp.py --out runs/r --depths 2 --depth-weights 1,0.5
--epochs 1 --lr 3e-5 --micro-tokens 4096` (~30 min/epoch at 4k tok/s), then `export_mtp.py`.
The trainer reproduces vLLM's drafter to 1% (checked by replaying captured drafter calls
with their KV history) so its numbers are trustworthy; use `--eval-only` first, response
tokens only, true-token criterion.

## Notes that cost time

- `capture.py` aligns hidden states by vLLM's request ids, which are `"<counter>-<uuid>"`
  in 0.27, and by `input_batch.req_ids` order within a step.
- Decode-time hidden states differ from prefill ones by ~0.9% (fp16 recurrent state);
  training on either gives the same drafter.
- Positions right after a rejection are systematically harder: vLLM's per-position
  acceptance is measured there, so it sits ~5 points below a whole-sequence top-1 rate.
  The chain simulation in `train_mtp.py --eval-only` accounts for that.
- Greedy decoding with speculation is not bit-deterministic across drafter configs
  (verify batches of 5 vs 1 token round differently), so 8 prompts × 1k tokens has a
  ±3% spread on tokens/step. Repeat before believing a 2% difference.
