# Training: the LOSO protocol

Five stages inside one leave-one-subject-out fold, with a gate between stage 1 and
stage 2 that decides whether the policy is ready for reinforcement learning at all.
One `LOSORunner.run` call covers all of them.

```
for each subject s in the dataset:
    test pool  = { s }
    train pool = every other subject                     <- the whole rest, N-1 subjects

    stage 0   fine-tune the CLIP engine on the train pool, save the fold checkpoint
    calib     fit (tau_hi, tau_lo) on the train pool      (only with --calibrate)

    repeat up to max_sft_rounds:
        stage 1   SFT on the train pool                  (<= sft_max_epochs, early stop)
        gate      four sufficiency judgements
        if decision != continue_sft: break

    stage 2   RFT: sample k, keep the passing ones, write them back as QA, refit
              (+ the corpus-level augmented sweep, filtered to this fold's pool)
    stage 3   GRPO for rl_total_steps steps              (only if decision == rl_ready)
    stage 4   one evaluation on subject s, optionally with test-time augmentation
```

**Stage 0 is in the chain, not beside it.** Every later stage consumes the
representation the CLIP engine produces, so running it as a separate command makes the
ordering a convention rather than a guarantee — and the failure mode (a policy trained
against a *stock* CLIP while the report says "fine-tuned") is silent and produces
plausible numbers. `train-clip-localiser` still exists for running the stage alone.
Set `training.clip_finetune: false` to skip it; `clip_required: true` makes a failure
fatal instead of a silent fall-back to the analytic front end.

**Stages 0–3 never see subject `s`.** Not its videos, annotations, QA rows, augmented
pairs, or CLIP weights — the checkpoint records the subjects it saw and asserts `s` is
absent before stage 1 starts. Subject `s` appears exactly once, in stage 4, under
inference only: TTA takes `tta_samples` draws and **no gradient step**, so `s` still
reaches no loss term. `pass_at_1` is always reported beside `pass_selected`.

Everything below is configured under `training:` in `configs/mewm_agent.yaml` and typed in
`mewm/config.py::TrainingConfig`.

---

## 1. What LOSO means here

**Per dataset, one fold per subject.** CASME^2 (`casme_sq`) has 22 subjects, so 22 folds.
For fold `s15`, subject `s15`'s videos are the test set and the remaining 21 subjects'
videos are the training pool. Both SFT and RL run on that same pool — this is the point
worth being explicit about, because "SFT on the non-test subjects" and "RL on the non-test
subjects" describe the *same* set of subjects, not two nested splits.

Subject-disjointness is the only split that means anything for micro-expressions: the same
subject's face, resting appearance and idiosyncratic AU habits recur across their clips, so
a random clip-level split leaks identity and inflates every number. `FoldSpec.check_disjoint()`
asserts this and is called both at construction and again at the top of `run_fold`.

```bash
python -m mewm.cli.main folds --dataset casme_sq
```

### The in-sample caveat

The configured default is `n_val_subjects: 0` — all N-1 non-test subjects train, and none
is held back. That is the protocol as specified, and it maximises training data, but it has
a consequence that every report carries explicitly:

> **The sufficiency gate is measured on the training pool.** A loss plateau therefore
> indicates the pool has been memorised, not that the policy has converged; and pass@1 is
> optimistic because the prompts were trained on.

Every `SufficiencyReport` sets `in_sample=True` and `measured_on="training_pool"`, appends
the caveat to `reasons()`, and `SFTOutcome.evaluated_on` records the same. The alternative
is one line of config:

```yaml
training:
  n_val_subjects: 1     # carve a validation subject out of the pool; gate on it instead
```

Fold construction, the gate and the stages are otherwise identical. The validation subject
is drawn with a seed derived from `(fold_seed, dataset, test_subject)`, so fold `s16`'s
choice does not depend on whether fold `s15` ran first, and a rerun reproduces it.

---

## 2. Stage 1 — SFT

`sft_max_epochs: 100` is a **ceiling, not a target**. Inside it, `SFTTrainer` early-stops on
`sft_patience` epochs without an improvement greater than `sft_min_delta`, and reports
`selected_epoch`, `stopped_early` and `stop_reason`. A fold that converges at epoch 40 stops
at 40; running the remaining 60 epochs would only overfit a pool that is already memorised.
The schedule is linear warm-up (`sft_warmup_ratio`) into cosine decay, matching stage 0.

Loss is masked to the assistant turn only:

```python
input_ids = prompt_ids + target_ids
labels    = [-100] * len(prompt_ids) + target_ids
```

Without the mask the model spends most of its gradient learning to reproduce the prompt,
which is both wasted and actively harmful to instruction following.

```bash
python -m mewm.cli.main sft --dataset casme_sq --fold s15 --dry-run   # schedule only
python -m mewm.cli.main sft --dataset casme_sq --fold s15             # needs a GPU
```

The policy must be **open-weight**. SFT and GRPO both need gradients, and the hosted
endpoints expose none, so `require_open_weight()` rejects a hosted id up front with an
explanation rather than letting the run fail at the first optimiser step. Hosted models
remain useful as frozen critics and as inference baselines.

---

## 3. The gate — four sufficiency judgements

`mewm/training/diagnostics.py`. The decision is `continue_sft` / `rl_ready` / `saturated`.

### 1a. Format alignment — can it hold the contract?

Share of sampled outputs that parse as a JSON object *and* satisfy the answer contract.
Threshold `format_pass_rate: 0.95`. This one is **absolute and checked first**: RL on
unparseable samples optimises the parser, not the reasoning. Parse failures and contract
failures are counted separately, because they need different fixes — a parse failure is a
decoding or prompt problem, a contract failure is a content problem.

### 1b. Loss plateau — has it hit a bottleneck?

Judged on the trailing `plateau_window` fraction of the curve, by **two** statistics:

* least-squares slope, `|slope| <= plateau_max_slope`, and
* coefficient of variation, `cv <= plateau_max_cv`.

Both are required. Slope alone would call an oscillating curve converged — a loss swinging
±0.2 around a flat mean has slope ≈ 0 and is obviously still unstable, which is exactly the
"no longer fluctuating" condition this judgement exists to check. A separate
`still_descending` flag distinguishes "still improving, let it run" from "flat but noisy".

### 2a. Headroom — is there room below the ceiling?

**"pass@k > pass@1" on its own proves nothing: it is a tautology.** pass@k is monotone
non-decreasing in k, so the inequality holds for every model that ever passes at all,
including a perfect one. `tests/verify_correctness.py` asserts this monotonicity so the
point cannot quietly be forgotten.

What is informative is the **gap plus a floor**:

| condition | verdict | meaning |
| --- | --- | --- |
| `pass_k < pass_at_k_floor` | `undertrained` | cannot pass even with k tries — RL has nothing to rank |
| `gap < pass_gap_min` | `saturated` | already ranks its passing answer first — RL has nothing to fix |
| otherwise | `rl_ready` | can pass, but not on the first try — precisely what RL fixes |

pass@k uses the unbiased estimator `1 - C(n-c, k)/C(n, k)`, evaluated as a product for
numerical stability, on `pass_at_k_samples: 16` draws per prompt.

### 2b. Reward distribution — three ways, not two

Sample many answers per question and look at the spread *per group*:

| group | verdict | action |
| --- | --- | --- |
| high and low mixed, `std >= reward_std_min`, mean in band | `spread` | trainable — start RL |
| all low | `undertrained` | SFT is not there yet — keep training |
| all high | `saturated` | nothing left to prefer — drop the prompt |

The **all-high** case is easy to omit, and omitting it is a real error: GRPO's advantage is
`A_i = (R_i - mean(R)) / std(R)`, so a group at the ceiling has an identically zero
advantage and contributes exactly as little as a group at the floor. A step taken on it is a
no-op that still costs a forward and backward pass. The same applies to a tight cluster
anywhere in the middle, which is why `classify_group` requires *both* usable spread and a
mean inside `[reward_mean_low, reward_mean_high]`.

The pool verdict is decided on the **majority of groups**, not the pooled mean: a pool of
degenerate groups, half at the floor and half at the ceiling, has a perfectly reasonable
pooled mean and trains nothing.

### The composed decision

Format first and absolute; then a descending loss blocks. Beyond that, `undertrained` from
either 2a or 2b sends the run back to SFT, and `rl_ready` from either is enough to proceed —
they measure the same readiness from two directions, and requiring both would stall on
whichever is noisier. `stage_rates` reports format/temporal/label/joint pass rates
separately so the bottleneck is visible instead of aggregated away.

```bash
python -m mewm.cli.main sft-report --run runs/sft/casme_sq/fold_s15/outcome.json --n 16 --c 4
```

---

## 4. Stage 2 — RFT, and why there is no DPO

The original plan was to "DPO or rank the candidates against the original QA pairs, since
DPO filters out inferior data". Those are two separable things, and only one of them is
wanted here:

* **DPO the objective** is a pairwise preference loss over `(chosen, rejected)`. It needs a
  reference policy, and it optimises a *margin* — it can raise the chosen answer's relative
  probability while lowering its absolute probability. That is the wrong target when the
  goal is "make the first generation the passing one".
* **DPO the filter** — discarding inferior generations — is the part that was actually
  wanted, and it needs no preference loss at all.

So the objective is removed and the filtering is kept, in
`mewm/training/candidate_filter.py`. A rejected candidate never enters a loss term; it
simply does not become a training row. What remains is a plain likelihood on material
already known to pass, which is standard rejection-sampling fine-tuning (STaR / RAFT) and is
a **supervised** step, not a reinforcement one.

Admission requires **both** conditions:

```python
accepted = candidate.passed and candidate.reward >= rft_min_reward
```

and ranking inside a prompt is `(passed, reward)` descending — a passing candidate always
outranks a higher-reward failing one, so the `rft_accept_top_k` budget can never be spent on
failures while a pass sits below the line.

---

## 5. Augmented QA pairs

Accepted candidates are written back to the dataset's own QA directory:

```
Q-T-A/<dataset>/augmented/fold_<subject>/
    <dataset>_me_lvqa_aug.jsonl            <- the 4-field rows, nothing else
    <dataset>_me_lvqa_aug_manifest.json    <- provenance
```

The fold is already in the directory name, so it is not repeated in the file name; the
`me_lvqa` stem matches the reference sets these rows are appended alongside.

### The format is exact

Four fields, in order, and nothing more:

```json
{"video_id": "casme_sq_augs15_s16_0102_1", "video": "s16_0102", "question": "...", "answer": "..."}
```

`validate_schema()` rejects any key-set or key-order mismatch and any answer that is not a
string or number, on both the write and the read path. This runs against the reference
format rather than against a description of it, which is what keeps an augmented file
loadable by the ordinary `load_qa_set` path.

### Provenance without breaking the format

The requirement to state which sample video each pair came from is in tension with a strict
4-field schema. Resolved by putting it in the two places that cost nothing:

* `video` is already the source long video's key, and `video_id` encodes
  `<dataset>_aug<fold>_<video>_<n>` — mirroring the reference `casme_sq_train_s15_..._1`
  shape, so the id is informative to a reader and parseable by `_suffix_index`;
* the sidecar manifest carries subject, event index, interval, reward, accepting reason and
  policy model, per pair.

### Two distinct leaks, both checked

1. **Source-subject leakage** — a pair generated from a video the fold is about to be tested
   on. `write_augmented(..., allowed_videos=fold.pool_videos())` **raises** on a stray video
   rather than dropping the row: a silent drop leaves a smaller file and no indication that
   the sampling stage was misconfigured.
2. **Cross-fold leakage** — fold `s15` loading the file fold `s16` wrote. These files
   outlive the run that produced them, so this is the one that bites months later.
   `load_augmented(..., allowed_videos=...)` raises on the way in.

Validation is on **video keys, not parsed subject ids**, deliberately. Deriving a subject
from a video string differs per corpus — SAMM's long videos are a flat clip layout, and
`datasets.py`'s `folder_rel.split("/")[0]` is wrong there — and a parsing mistake would turn
the leak check into a silent no-op. A video key comparison is exact.

### They are kept separable from annotation

`QASet.augmented_ids` tracks which rows are machine-generated; `annotated_items()` and
`augmented_items()` split them. Augmented pairs are **off by default** in `load_qa_set` and
only the training driver asks for them, so no evaluation or report can quote a generated
pair as if it were reference annotation.

```bash
python -m mewm.cli.main build-augmented-qa --dataset casme_sq        # inspect + leak-check
```

---

## 6. Stage 3 — GRPO

`rl_total_steps: 1000` counts **optimisation steps**, not passes over the prompt set. Each
step draws `rl_group_size: 8` samples for each of `rl_prompts_per_step: 4` prompts, so a
1000-step run is 32 000 generations and 1000 updates.

Two properties of the loop worth knowing:

**Prompt admission.** With `rl_admit_prompts: true`, a group whose reward std falls below
`reward_std_min` is skipped and counted rather than stepped on, and the reason names which
end it degenerated at (`floor` or `ceiling`). `n_skipped` appears in every step record —
if it stays high, the gate let the run through too early.

**Prompt order is permuted.** The prompt list is reshuffled each time a pass over it is
exhausted. Walking a fixed order for 1000 steps would pair the same prompts into the same
batches on every pass, correlating the gradient with an arbitrary property of the list
order.

The reward is the five-dimensional composite of eq. (11) under the three-stage cosine
curriculum, with `R_causal` supplied online by the **frozen** world-model judge. `R_temp`
reads the same `evaluation.iou_threshold` the metrics do, so a policy cannot be trained
against one criterion and scored against another.

---

## 7. Running it

```bash
# Fold arithmetic and gate wiring, no GPU:
python -m mewm.cli.main loso --dataset casme_sq --folds s15,s16 --dry-run

# Overrides:
python -m mewm.cli.main loso --dataset casme_sq --policy Qwen3-VL-8B \
                             --sft-epochs 100 --rl-steps 1000 --output runs/loso
```

**Start with a subset.** 22 folds × 3 datasets × (100 SFT epochs + 1000 RL steps) is a large
amount of compute, and a protocol bug found on fold 1 costs the same as a protocol bug found
on fold 22. `--folds s15,s16` runs two.

`LOSORunner` takes its policy backend, sampler and evaluator as injected callables, which is
what lets `--dry-run` exercise the entire fold-and-gate chain with no parameters at all —
and is how the fold arithmetic gets tested.

---

## 8. Tests

```bash
python tests/test_training_protocol.py    # 66 protocol tests
python tests/verify_correctness.py        # 125 checks, incl. the pass@k / advantage audit
python tests/test_invariants.py           # 57 structural invariants
```

`test_training_protocol.py` covers fold disjointness, the cross-fold leak, pass@k against
the closed form on every `(n, c, k)` up to 16, plateau behaviour on synthetic curves
(including the oscillating one), the three-way reward verdict including all-high, the 4-field
schema, the filter's pass-before-reward ordering, and that the DPO objective stays removed.
That last check matches on `\bDPO\b` — a plain substring search also matches `ENDPOINT`,
which is how it produced a page of false positives the first time it was done by eye.

---

## 6. CLIP dual-tower localiser (引擎 CLIP 化, 修改方案 §1/§2)

The analytic V2 slot readout / V3 latent / M1 prediction curve are replaced at
inference by a **trained** dual tower when a fold checkpoint exists:

```bash
# Train one fold (pool = every subject except the held-out one; no LLM calls):
python -m mewm.cli.main train-clip-localiser --dataset casme_sq --subjects s15

# All folds:
python -m mewm.cli.main train-clip-localiser --dataset casme_sq
```

Architecture: local CLIP ViT-B/16 (`clip.weights_path`), last **4 vision** and
**2 text** encoder layers fine-tuned, everything else frozen. RGB frames feed the
vision tower; the deterministic optical-flow motion descriptions
(`mewm/engines/motion_description.py`) feed the text tower. Losses (all local
gradients — the hosted APIs appear nowhere in this path):

```
loss = lambda_align * InfoNCE(v, m)                 frame <-> motion description
     + lambda_loc   * focal BCE                     GT intervals from the QA pairs
     + lambda_cont  * head-motion triplet           head movement as an explicit negative
     + lambda_prop  * (1 - soft IoU)                dense boundary gradient (方案 §3)
     + lambda_distill * MSE(transition head, analytic activations)
```

Checkpoints land in `runs/clip_localiser/<dataset>/fold_<subject>/clip_localiser.pt`
with the training subjects recorded; `run`/`spot` pick them up automatically
(`--clip-localiser auto`, the default) and refuse a checkpoint that saw the test
subject. `--clip-localiser off` restores the analytic curve.

SFT additionally carries `training.sft_prop_weight` (lambda_prop, 方案 §3): extra CE
weight on the onset/offset/apex number tokens of `part1_proposals`, so localisation
and analysis tokens share one backward pass. It only applies to gradient-capable
(local) policies; a hosted policy is rejected by `require_open_weight` as before.
