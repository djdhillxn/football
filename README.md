# Failure-Directed Robust Multi-Agent Robot Soccer

This repository studies cooperative high-level skill selection for small-sided robot soccer.
Phase 2 tested two attackers against one scripted defender; Phase 3 adds calibrated 2v2/3v2
team-play scenarios, padded roster masks, a parameter-shared GRU actor, and a recurrent centralized
critic. Policies train in an explicit kinematic simulator and freeze before evaluation in an
independently integrated Pymunk rigid-body simulator.

The project is **multi-fidelity, sim-to-real-oriented robustness research**. Pymunk evaluation is
sim-to-sim transfer and a controlled proxy for parts of the sim-to-real gap; it is not evidence of
physical-robot deployment. The completed three-seed Phase 2 result is a failed overall gate:
failure-directed randomization improves Pymunk profile/grid robustness, but loses too much nominal
abstract competence and learns no stable passing. Phase 3 is implemented but has not been fully
graduated: seed-3 Stages A--D learned substantial cooperation, but historical Gate B failed
Pymunk 2v2 open play at 0.49 versus the locked 0.55 floor. The focused Stage-R repair is
implemented and smoke-tested but scientifically unrun. CC-FDR and final seeds remain locked.

## Contribution and architecture

The proposed method periodically evaluates a frozen actor on named perturbation profiles, smooths
each profile's failure rate, and raises its future sampling probability subject to a uniform floor
and a hard probability cap. This directs finite training effort toward observed weaknesses without
letting one failure mode monopolize training.

```text
 attacker 0 local observation ─┐
                               ├── shared decentralized actor ── seven macro-actions
 attacker 1 local observation ─┘                    │
                                                    ├── abstract simulator (training)
 true joint state ── centralized critic (MAPPO) ───┘
                                                    └── frozen actor
                                                         │
                                                         └── Pymunk rigid-body simulator
                                                             (zero-shot transfer evaluation)
```

IPPO replaces the centralized critic with a parameter-shared local critic. The actor never sees
the global state or hidden perturbation parameters. Both methods select exactly seven skills:
approach, dribble left, dribble right, shoot, pass, support, and hold/face ball.

The RoboCup-facing endgame is deliberately hierarchical:

```text
learned high-level football policy
        ↓
robot perception/localization
        ↓
walking / turning / dribbling / kicking primitives
        ↓
robot platform
```

This repository demonstrates high-level CTDE MARL under partial observability, recurrent
decision-making, cooperative passing/positioning, procedural adversaries, sim-to-sim transfer,
reward discipline, and gated experimentation. It does not train NAO/humanoid joint control, and
success here alone is not a claim of winning RoboCup. This boundary is consistent with recent
UW–Madison work on [multi-robot RL through abstract simulation](https://arxiv.org/abs/2503.05092)
and [RL integrated within a classical robot-soccer stack](https://arxiv.org/abs/2412.09417);
our simulator and deployment evidence are not the same as theirs.

## Repository structure

```text
configs/              Inherited experiment configurations and perturbation profiles
robosoccer/           Configuration, environments, training, evaluation, and utilities
scripts/              Thin module-based command-line entry points
tests/                Phase 1/2 and recurrent Phase 3 correctness tests
notebooks/            Three Colab/VS Code command dashboards
reports/              Paper-style report, technical ledger, bibliography, generated results
webots/README.md      Optional future adapter contract; no untested Webots world is included
```

## Installation

Python 3.10–3.13 is supported. The pinned set was selected against current official compatibility:
PyTorch 2.7 requires Python 3.10+, PettingZoo 1.26 uses Gymnasium 1.x, and Pymunk 7.3 supports
modern CPython wheels.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Fast validation

```bash
ruff check .
pytest -q
python -m scripts.train --config configs/smoke_test.yaml
```

The smoke configuration performs several tiny MAPPO updates on CPU, exercises failure-directed
sampling, saves complete checkpoints and TorchScript actors, writes CSV metrics, and verifies
headless rendering. It is an infrastructure check, not a scientific experiment.

The recurrent Phase 3 smoke path is:

```bash
python -m scripts.calibrate_phase3 --config configs/phase3_base.yaml \
  --output-dir runs/phase3_calibration_smoke --episodes 20 --smoke
python -m scripts.benchmark_throughput --config configs/phase3_smoke.yaml \
  --num-envs 4 8 16 --updates 2
python -m scripts.train --config configs/phase3_smoke.yaml
```

The calibration smoke validates code and the expected ordering but deliberately sets
`training_authorized=false`. Only the non-smoke 100-episode calibration can satisfy Gate A.

## Phase 3 gated workflow

Phase 3 has four nominal stages with cumulative target budgets configured in YAML: 0.5M-step 2v2
open play, 1.0M pass-required play, 1.5M mixed 2v2/3v2, and 2.0M open/press team play. Every stage
can resume from the preceding checkpoint. A scientific run requires the saved full calibration:

```bash
python -m scripts.calibrate_phase3 --config configs/phase3_base.yaml \
  --output-dir runs/phase3_calibration_seed3 --episodes 100

python -m scripts.train --config configs/phase3_recurrent_nominal.yaml \
  --stage stage_a --seed 3 \
  --calibration-summary runs/phase3_calibration_seed3/calibration_summary.json
```

Stages B--D use `--resume` with the preceding best nominal checkpoint. CC-FDR starts from the best
completed, graduated nominal policy. Historical Stage D did not graduate, so Stage R must run
first. It warm-starts weights and normalizers from the exact Stage-D actor while resetting
optimizers, schedule, rollout state, and Stage-R counters:

```bash
python -u -m scripts.evaluate_phase3 \
  --run-dir runs/20260723_094519_phase3_recurrent_nominal_mappo_seed3 \
  --checkpoint best_nominal --episodes 100 --seed-base 360000 \
  --stage-r-r0-audit
python -u -m scripts.evaluate_phase3_gates --gate reward-invariants \
  --config configs/phase3_stage_r.yaml \
  --output runs/logs/stage_r_reward_invariant_pretraining.json
python -u -m scripts.train --config configs/phase3_stage_r.yaml --seed 3 \
  --num-envs 128 --total-steps 500000 \
  --warm-start runs/20260723_094519_phase3_recurrent_nominal_mappo_seed3/models/best_nominal_checkpoint.pt
```

Gate B requires 2v2 open success at least 0.55, 3v2 open success at least 0.35,
pass-required cooperative success at least 0.30, and median successful sequences of at least eight
seconds. Gate B-R preserves those checks and adds style, cooperation, 3v2, reward-alignment, and
reward-invariant safeguards on untouched seed base 370000. Only its explicit
`passed=true`/`cc_fdr_authorized=true` artifact permits:

```bash
python -u -m scripts.train --config configs/phase3_cc_fdr.yaml --seed 3 \
  --warm-start runs/<stage-r-run>/models/best_stage_r_checkpoint.pt \
  --authorization-artifact runs/<stage-r-run>/eval/phase3_gate_b_r.json \
  --calibration-summary runs/phase3_calibration_seed3/calibration_summary.json
```

Gate C requires profile-mean improvement of at least 0.10, nominal loss no more than
0.10, grid regression no more than 0.05, and no cooperation regression. Development uses seed 3;
new final seeds 4, 5, and 6 remain disabled until all gates pass.

PettingZoo API checks can be isolated with:

```bash
pytest -q tests/test_core.py -k parallel_api
```

## Baselines and training

```bash
python -m scripts.evaluate_baselines --config configs/base.yaml --episodes 100

python -m scripts.train --config configs/ippo_nominal.yaml
python -m scripts.train --config configs/mappo_nominal.yaml
python -m scripts.train --config configs/mappo_uniform_dr.yaml
python -m scripts.train --config configs/mappo_failure_dr.yaml
```

Focused ablations use ordinary YAML-valued dot overrides:

```bash
python -m scripts.train \
  --config configs/mappo_failure_dr.yaml \
  --run-name mappo_failure_no_action_delay \
  'randomization.disabled_parameters=[action_latency]'
```

Other useful overrides include `experiment.seed=1`, `train.total_steps=2000000`, and
`observations.expose_perturbations_to_critic=false`. Full checkpoints preserve optimizer,
normalizer, RNG, learning-schedule, best-score, and curriculum state:

```bash
python -m scripts.train --config configs/mappo_failure_dr.yaml \
  --resume runs/<run>/checkpoints/checkpoint_step_200000.pt
```

## Learned-policy evaluation

```bash
RUN_DIR="$(cat runs/latest_mappo_failure_dr.txt)"

python -m scripts.evaluate --run-dir "$RUN_DIR" --checkpoint best \
  --simulator abstract --suite standard --episodes 100
python -m scripts.evaluate --run-dir "$RUN_DIR" --checkpoint best \
  --simulator pymunk --suite transfer --episodes 100
python -m scripts.evaluate --run-dir "$RUN_DIR" --checkpoint best \
  --simulator abstract --suite profiles
python -m scripts.evaluate --run-dir "$RUN_DIR" --checkpoint best \
  --simulator pymunk --suite robustness
python -m scripts.evaluate --run-dir "$RUN_DIR" --checkpoint best \
  --simulator pymunk --suite cooperation --seed 250000 \
  --prefix confirmatory_pymunk_cooperation
```

Standard evaluation covers all three defender modes. Profile evaluation forces every configured
perturbation family. The robustness suite sweeps action delay against localization noise and saves
long-form data, pivot tables, and Matplotlib heatmaps. Running abstract standard and Pymunk transfer
evaluation for the same learned run also creates transfer-gap metrics.

The cooperation suite uses a mirrored pass-needed initial state: the carrier's direct goal lane is
blocked while its teammate has an open forward lane. Reported counts distinguish opportunities,
pass choices, valid attempts, completions, receiver possession, and post-pass goals. Other metrics
include return distribution, success, time to score, possession, redundant chasing, collisions,
invalid actions, worst-decile return/CVaR, profile minima, robustness area, bootstrap intervals,
and canonical cross-simulator gaps.

Audit delayed-action semantics and record policy-dependent matched traces with:

```bash
python -m scripts.audit_action_delay --config configs/base.yaml \
  --output-dir runs/logs/phase2_protocol --maximum-latency 5
python -m scripts.trace_action_delays --run-dir "$RUN_DIR" \
  --simulator pymunk --delays 0 1 2 3 4 5 --seed 260000
```

## Videos

```bash
python -m scripts.record_video --run-dir "$RUN_DIR" --checkpoint best \
  --simulator abstract --episodes 3
python -m scripts.record_video --run-dir "$RUN_DIR" --checkpoint best \
  --simulator pymunk --episodes 3
python -m scripts.record_video --run-dir "$RUN_DIR" --checkpoint best \
  --simulator pymunk --episodes 3 --scenario cooperation --matched --seed 250000
python -m scripts.record_video --config configs/base.yaml --baseline role_based \
  --simulator abstract --episodes 1

python -u -m scripts.record_phase3_video --run-dir runs/<phase3-run> \
  --checkpoint best_stage_r --simulator pymunk \
  --scenario phase3_2v2_open --defender-style predictive \
  --seed 370123 --seed-category gate --until-terminal
```

Rendering uses Pillow RGB arrays and imageio-ffmpeg, so no display server is required. Videos are
16:9 MP4s with simulator, seed, profile, actions, method, and final outcome overlays.
The Phase-3 recorder also supports exact forced-style replay, fixed-length clips with
`clip_end_metrics`, and terminal/full-match mode. Its manifest merges and deduplicates by
checkpoint/backend/scenario/style/profile/seed/mode, preserving earlier valid records.

## Artifacts and aggregation

Each command writes a timestamped run with a resolved configuration and auditable metadata:

```text
runs/YYYYMMDD_HHMMSS_experiment_method_seed0/
├── resolved_config.yaml         ├── eval/
├── run_metadata.json            ├── plots/
├── checkpoints/                 ├── videos/
├── models/                      └── logs/
│   ├── best_checkpoint.pt           ├── metrics.csv
│   ├── final_checkpoint.pt          ├── curriculum_history.csv
│   ├── best_actor.ts                └── tensorboard/
│   └── final_actor.ts
```

`runs/latest_<experiment>.txt` points to the latest complete run. New local pointers use portable
`runs/<run-directory>` values; readers also recover the basename of historical absolute
Colab/Mac paths. `python -m scripts.audit_workspace` reports stale pointers, incomplete metadata,
missing referenced artifacts, and notebook-only runs without mutating the workspace.
`runs/experiment_manifest.jsonl` is the discovery ledger. Compare explicit runs or completed final
runs with:

```bash
python -m scripts.compare_runs --phase final --export-report
python -m scripts.compare_runs --runs runs/run_a runs/run_b runs/run_c --export-report
```

Aggregation preserves the exact evaluation-suite name and training seed. It creates raw and
suite-level CSV/JSON, canonical same-run abstract-intercept/Pymunk-transfer gaps, a seed-level
replication gate, suite-specific plots, and `reports/generated_results.tex`. Transfer, profile,
robustness, and cooperation suites are never pooled into one error bar.

### Colab, Google Drive, and Mac synchronization

Code stays in Git, while finished experiment artifacts use one persistent Drive project:

```text
/content/robot-soccer-transfer/runs ──full immediate push──> MyDrive/RobotSoccerTransfer/runs
             ^                                                        |
             |                                                        |
             +────────── full Colab restore ─────────────────────────+
                                                                      |
                       analysis-only Mac pull <───────────────────────+
                                  |
                                  +──> <football checkout>/runs

MyDrive/RobotSoccerTransfer/reports ───────────────> <football checkout>/reports
```

Both notebooks explicitly pull the complete artifact workspace, including checkpoints needed for
evaluation and resume, immediately after initialization. Every training cell then saves its newly
completed run before the cell finishes. Failed runs are also
saved after their metadata reaches `failed`, so their detailed logs survive for diagnosis.
Evaluation and video cells re-save the modified run, while comparison and report-build cells save
their outputs in the same cell. The final notebook sync remains a safety net, not a required step.

With Google Drive for desktop running on macOS, merge Drive artifacts directly into this checkout
with:

```bash
python -m scripts.sync_drive_artifacts
```

The command auto-detects a single
`~/Library/CloudStorage/GoogleDrive-*/My Drive/RobotSoccerTransfer` folder. It can be pointed at a
specific location when Drive uses another account or the folder has not been auto-detected:

```bash
python -m scripts.sync_drive_artifacts pull \
  --drive-project "/path/to/My Drive/RobotSoccerTransfer"
```

`ROBOSOCCER_DRIVE_PROJECT` provides the same override. Use `--dry-run` to preview a pull. The Mac
default is an analysis-only, non-destructive merge: it keeps configurations, metadata, JSON/CSV,
training and TensorBoard logs, plots, and videos, while pruning `models/` and `checkpoints/` from
the Drive traversal and skipping `.pt`, `.pth`, `.ts`, `.ckpt`, `.pkl`, `.pickle`, and `.zip`
payloads.
Those files remain safely on Drive and Colab-to-Drive saves are always full. The pull preserves
local-only artifacts, skips `running` runs by default, restores comparisons and manual videos, and
rebuilds local `latest_*.txt` pointers and the manifest from metadata. Authored report sources
(`main.tex`, `surrogate_notes.tex`, and `references.bib`) remain Git-controlled and cannot be
overwritten by an older Drive copy;
generated LaTeX, figures, and PDFs are synchronized into root `reports/`.
Generated report artifacts use a newest-file-wins rule so an older Drive report cannot replace a
newer local build (and a stale Colab checkout cannot replace a newer Drive result).

The first pull walks only lightweight trees; subsequent pulls also use a local run-signature cache
and up to eight parallel copies. To remove heavyweight artifacts downloaded before this policy,
without changing Drive, run:

```bash
python -m scripts.sync_drive_artifacts pull --prune-local-training-artifacts
```

An explicit full local restore remains available when local model execution is genuinely needed:

```bash
python -m scripts.sync_drive_artifacts pull --include-training-artifacts
```

The notebook helpers use the same implementation. The equivalent explicit commands are:

```bash
python -m scripts.sync_drive_artifacts push-run --run-dir runs/<finished-run>
python -m scripts.sync_drive_artifacts push-all
```

`push-run` accepts only `complete` or `failed` metadata, copies metadata last, and writes portable
Drive pointers such as `runs/<run-directory>`. Pulls use the immutable-run convention and compare
existing files by size; analysis-only pulls never descend into checkpoint/model directories.
Pushes content-check mutable text before updating Drive. Use `--verify-text` on a pull when an
existing text artifact may have been rewritten in place; this stricter audit can be slower because
Drive must hydrate those files.
Sync activity is recorded in `runs/logs/artifact_sync.log`, in addition to each experiment's own
training, evaluation, video, CSV, and TensorBoard logs.

## Experiment matrix

| Method | Critic | Training distribution | Required role |
|---|---|---|---|
| Random / double chase / role based | none | none | scripted baselines |
| IPPO nominal | shared local | nominal | decentralized-value baseline |
| MAPPO nominal | centralized team | nominal | no-randomization control |
| MAPPO uniform DR | centralized team | uniform profiles | randomization control |
| MAPPO failure-directed DR | centralized team | observed failures | proposed method |
| Failure DR minus delay / communication / observation | centralized team | ablated profiles | mechanism tests |

Use multiple seeds for the final matrix. Training, curriculum validation, ordinary validation,
abstract test, Pymunk transfer, and video scenarios use disjoint configurable seed ranges.

## Colab workflow

The three notebooks are command dashboards rather than hidden implementations:

- `notebooks/phase1_environment_and_baselines.ipynb`: initialization, tests, smoke run, baselines,
  IPPO, transfer, explicit readiness audit, video, and Drive sync.
- `notebooks/phase2_training_transfer_report.ipynb`: MAPPO variants, ablations, full evaluation,
  comparison, reports, and Drive sync. It recomputes the Phase-1 audit before allowing training.
- `notebooks/phase3_adversarial_teamplay.ipynb`: calibration, throughput, staged recurrent
  training, the self-contained Stage-R fast path in Sections 25–38, guarded CC-FDR, Gate B/B-R/C,
  videos, reports, and Drive sync. Executed A–D outputs are preserved; expensive final seeds are
  disabled by default.

They keep the repository under `/content`, use Drive only for persistent artifacts, refuse to pull
over a dirty checkout, restore the complete artifact workspace, and allow later experiment sections
to run after initialization without replaying earlier cells.

## Reports

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=reports reports/main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=reports reports/surrogate_notes.tex
```

The main report is a paper-style account. The surrogate report is the technical ledger for exact
settings, attempts, failures, and decisions. Both place their build products in `reports/` and
clearly separate completed Phase-1/2 and Phase-3 A–D evidence from implemented-but-unrun Stage R.

## Current status, limitations, and expected deliverables

The repository implements the original environments and feed-forward comparisons plus the Phase 3
fixed-roster tasks, recurrent MAPPO, competence-constrained failure direction, calibration and
development gates, portable artifact audit, throughput instrumentation, upgraded video path, and
an independently runnable notebook. Stage R additionally implements reward schema 2, a
possession-chain progress high-water mark, forced-style R0, Gate B-R, isolated seed streams,
warm-start/reset semantics, and fail-closed CC-FDR authorization. This statement concerns software
availability, not Stage-R performance: R0, the 500k run, Gate B-R, CC-FDR, and final seeds are
unrun.

The task remains deliberately reduced: vector observations, two or three attackers, scripted
defenders, high-level skills, and synchronous environments. It excludes learned locomotion,
images, mandatory self-play, 5v5 tactics, ROS, Webots claims, and physical validation. Transfer
success in Pymunk shows robustness to a controlled fidelity change, not readiness for a humanoid
robot. Physical work requires perception, localization, collision-safe locomotion, kicking,
emergency stops, and hardware validation.

A bounded future bridge to RoboCup 2D, Webots, or a humanoid stack becomes appropriate only after
the hard calibration, pass-required cooperation, recurrent nominal, competence-preserving
robustness, and sustained 3v2 video gates all pass. No large external football framework is a
Phase 3 dependency.

Expected final deliverables are multi-seed run directories, frozen TorchScript actors, baseline and
transfer tables, robustness grids, representative videos, generated report figures, and compiled
PDFs. Webots is a later fidelity tier only after the mandatory matrix is complete.

## Prioritized execution roadmap

Phase 1, Phase 2, Gate A, Stages A–D, and historical Gate B are complete. Do not rerun A–D
initially. Execute notebook Sections 25–38:

1. R0: restore Stage-D `best_nominal` and run the frozen 4×2×2 audit.
2. R1: run and save the reward-invariant checks.
3. R2: train exactly one 500k Stage-R seed-3 run.
4. R3: run the held-out, abstract-only validation matrix.
5. R4: run Gate B-R on untouched seeds.
6. R5: record matched full-match success/failure videos.
7. R6: update and sync reports and the machine-readable summary.

If Gate B-R fails, stop and follow its recorded branch. If it passes, Section 39/CC-FDR becomes
authorized but does not run automatically. Gate C and seeds 4–6 remain later gated work.
