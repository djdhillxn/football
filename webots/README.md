# Optional Webots fidelity tier

Webots is intentionally not part of the mandatory pipeline. The tested core stops at zero-shot
transfer from the abstract trainer to the independent planar Pymunk simulator. Adding Webots too
early would mix asset, motion-controller, sensor, and simulator-version problems into the central
domain-randomization comparison.

A responsible adapter needs a current Webots installation and a verified official NAO or OP3
world with legally redistributable motion assets. It should load `models/best_actor.ts`, construct
the same 61-element local vector from supervisor/controller state, apply the saved observation
normalizer, and map the seven outputs to existing walk, turn, approach, dribble, kick, pass,
support, and hold primitives. It must run headlessly where supported, record action/observation and
outcome traces, and fail clearly when Webots or a required motion asset is unavailable.

Before calling Webots transfer successful, validate observation ordering and units, timestamp and
delay semantics, skill completion/interrupt behavior, goal and terminal detection, deterministic
seed handling, and at least one complete policy rollout in a launched world. A controller file or
world that has not launched is only a scaffold.

Physical deployment is a separate phase. It requires an existing calibrated perception and
localization pipeline, stable locomotion and kicking stacks, communication timestamps, fall and
collision handling, emergency stops, hardware-safe action limits, and logged robot trials. Neither
Webots startup nor Pymunk success justifies a physical sim-to-real claim.

Current status: **documented only; no Webots world, adapter, or rollout is claimed or tested.**

