# Research Idea: Scene-Grounded Humanoid Policies

Humanoid policies that act from a **persistent, static 3D scene memory** — not from per-frame perception.

## The problem

Feeding a policy per-frame monocular 3D (pointmaps, depth) gives it a world that *jitters*: scale drift and pose noise every frame. The robot never gets a stable notion of "where things are" — it reacts to snapshots instead of understanding a space.

## The idea

1. **Static world frame.** Maintain one persistent 3D reconstruction of the scene. The scene never jitters. The robot localizes *itself* inside it by fusing 2D-3D matching against the map with its own proprioception/ego-motion. Optically + sensor-grounded self-localization in a world that holds still.

2. **Policy consumes the map.** A scene-conditioned policy (map + current pose + goal → whole-body actions) acts in this stable frame. Spatial memory comes free: things that left the field of view are still in the map.

3. **Curriculum: complete scenes first, online reconstruction second.**
   - Stage 1: train the scene policy in *fully reconstructed* 3D scenes (privileged, complete maps). Learning to act is decoupled from learning to perceive.
   - Stage 2: at deployment the map is built *online* while moving — streaming reconstruction plugs into the same policy in the backend.

4. **Confidence-gated bootstrap.** Cold start: the map is empty. A tiny bootstrap policy moves the robot to create parallax/stereo; the reconstruction grows; the big scene policy takes over. The two are merged by **map confidence** (per-region observation coverage / uncertainty, queried along the intended trajectory) — the gate can be a small learned network, giving learned-vs-handcrafted gating as an ablation.

5. **Later — precision expert.** Volumetric scene memory caps out on high-accuracy tasks. The same confidence-fusion machinery extends to a third, image-based expert for fine manipulation. Not phase 1.

## Claims to test

- **Persistent map beats per-frame 3D** for whole-body policies (controlled ablation; nobody has shown this).
- **Complete-scene training transfers to online reconstruction** (privileged teacher → streaming student).
- **Confidence gating** beats either policy alone, and a learned gate beats a handcrafted one.
- **Spatial-memory probes**: tasks where the goal/object leaves the field of view.

## Why humanoid specifically

The camera bounces at gait frequency and the policy *causes its own viewpoints* — perception and control are coupled through the body. Action-aware mapping ("move to see") becomes a learnable behavior, not a heuristic.

## Sim-to-real path

Train in simulation with ground-truth depth/pose → degrade to estimated poses → swap in streaming RGB-only reconstruction. Real-world data enters as scanned scenes for training and as the online reconstruction at deployment.
