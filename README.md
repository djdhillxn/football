# Failure-Directed Robust Multi-Agent Robot Soccer

This repository studies cooperative high-level skill selection for two robot-soccer attackers
against one scripted defender. A parameter-shared policy is trained quickly in an explicit
kinematic simulator and frozen before evaluation in an independently implemented Pymunk
rigid-body simulator. The core question is whether failure-directed domain randomization can
improve zero-shot cross-simulator robustness to delays, perception errors, communication loss,
locomotion uncertainty, kick variation, and changed ball dynamics.

The project is **multi-fidelity, sim-to-real-oriented robustness research**. Pymunk evaluation is
sim-to-sim transfer and a controlled proxy for parts of the sim-to-real gap; it is not evidence of
physical-robot deployment. No final experimental result is claimed until completed run artifacts
have been generated and aggregated.

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

## Repository structure

```text
configs/              Inherited experiment configurations and perturbation profiles
robosoccer/           Configuration, environments, training, evaluation, and utilities
scripts/              Thin module-based command-line entry points
tests/test_core.py    Concentrated simulator, PPO, curriculum, artifact, and video tests
notebooks/            Two Colab/VS Code command dashboards
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
  'randomization.disabled_families=[action_delay]'
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
```

Standard evaluation covers all three defender modes. Profile evaluation forces every configured
perturbation family. The robustness suite sweeps action delay against localization noise and saves
long-form data, pivot tables, and Matplotlib heatmaps. Running abstract standard and Pymunk transfer
evaluation for the same learned run also creates transfer-gap metrics.

Reported metrics include return distribution, goal success, time to score, pass completion,
possession, redundant chasing, collisions, invalid actions, worst-decile return/CVaR, profile
minima, robustness area, bootstrap intervals, and cross-simulator gaps.

## Videos

```bash
python -m scripts.record_video --run-dir "$RUN_DIR" --checkpoint best \
  --simulator abstract --episodes 3
python -m scripts.record_video --run-dir "$RUN_DIR" --checkpoint best \
  --simulator pymunk --episodes 3
python -m scripts.record_video --config configs/base.yaml --baseline role_based \
  --simulator abstract --episodes 1
```

Rendering uses Pillow RGB arrays and imageio-ffmpeg, so no display server is required. Videos are
16:9 MP4s with simulator, seed, profile, actions, method, and final outcome overlays.

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

`runs/latest_<experiment>.txt` points to the latest complete run and
`runs/experiment_manifest.jsonl` is the discovery ledger. Compare explicit runs or completed final
runs with:

```bash
python -m scripts.compare_runs --phase final --export-report
python -m scripts.compare_runs --runs runs/run_a runs/run_b runs/run_c --export-report
```

Aggregation groups seeds by experiment name and creates comparison CSV/JSON, success and transfer
plots, and `reports/generated_results.tex`. The checked-in generated-results file is deliberately a
pending-results notice; it contains no fabricated numbers.

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

The two notebooks are command dashboards rather than hidden implementations:

- `notebooks/phase1_environment_and_baselines.ipynb`: initialization, tests, smoke run, baselines,
  IPPO, transfer, explicit readiness audit, video, and Drive sync.
- `notebooks/phase2_training_transfer_report.ipynb`: MAPPO variants, ablations, full evaluation,
  comparison, reports, and Drive sync. It recomputes the Phase-1 audit before allowing training.

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
clearly separate Phase-1 diagnostics from pending Phase-2 comparative results.

## Current status, limitations, and expected deliverables

The repository implements the mandatory abstract environment, independent Pymunk transfer
environment, parameter-shared IPPO/MAPPO, failure-directed curriculum, resumable artifact flow,
evaluation suites, videos, aggregation, notebooks, tests, and reports. This statement concerns
software availability, not final performance. Full multi-seed training remains to be run.

The task is deliberately reduced: vector observations, two attackers, scripted defenders,
high-level skills, synchronous environments, and feed-forward policies. It excludes learned
locomotion, images, recurrent memory, self-play, learned opponents, 5v5 tactics, ROS, Webots claims,
and physical validation. Transfer success in Pymunk would show robustness to a controlled fidelity
change, not readiness for a humanoid robot. Physical work requires perception, localization,
collision-safe locomotion, kicking, emergency stops, and hardware validation.

Expected final deliverables are multi-seed run directories, frozen TorchScript actors, baseline and
transfer tables, robustness grids, representative videos, generated report figures, and compiled
PDFs. Webots is a later fidelity tier only after the mandatory matrix is complete.

## Fifteen-day execution roadmap

1. **Days 1–2:** run local/Colab tests and smoke jobs; inspect rendered dynamics and baseline traces.
2. **Days 3–4:** run nominal IPPO and MAPPO pilot seeds; tune only from validation artifacts.
3. **Days 5–6:** run uniform-randomization pilots and verify sampled profile coverage.
4. **Days 7–8:** run failure-directed pilots; inspect probability entropy and cap behavior.
5. **Days 9–10:** lock hyperparameters and launch the planned multi-seed training matrix.
6. **Days 11–12:** run fixed abstract, profile, Pymunk transfer, and robustness evaluations.
7. **Day 13:** run delay/communication ablations and record qualitative success/failure videos.
8. **Day 14:** aggregate artifacts, audit seed separation, and compile both reports.
9. **Day 15:** reproduce key tables from restored Drive artifacts and document all failed runs.
