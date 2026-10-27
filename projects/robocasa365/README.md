# VLA Modeling Experiments and Failure Analysis on RoboCasa365

An implementation study of VLA policies and their failure modes on RoboCasa365:
**GR00T N1.7-3B atomic fine-tuning**, **SmolVLA-0.45B memory modeling**,
and instruction-following tests of the
[official leaderboard](https://robocasa.ai/leaderboard)'s **SOTA Awomo-0.5I (#1)**
and **Xiaomi-Robotics-1 (#5)** (October 28, 2026).

## Highlights

- Fine-tuned **GR00T N1.7-3B on atomic data**, achieving **50.0% Atomic-Seen success**.
- Implemented **SmolVLA-0.45B temporal memory** and analyzed lower success
  (**14.26% vs. 31.11%** for the base checkpoint), with no observed failure recovery.
- Tested **10 custom instructions per SOTA model** and investigated recurring
  instruction-following failures through related research.

## Setup

### Train

<details>
<summary>Install training dependencies</summary>

```bash
git clone https://github.com/siwooyong/lerobot.git /content/lerobot
cd /content/lerobot
git switch --track origin/feature/robot-learning

python -m pip install --upgrade pip wheel "setuptools>=71,<82"

python -m pip install \
  torch==2.11.0 \
  torchvision==0.26.0 \
  torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install \
  --no-deps \
  torchcodec==0.11.1 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e ".[training,groot]"
```

</details>

<details>
<summary>Optional authentication for Hugging Face access and W&B experiment tracking</summary>

```bash
hf auth login
wandb login
```

</details>

<details>
<summary>Download data and install FlashAttention</summary>

```bash
mkdir -p projects/robocasa365/data

hf download ember-lab-berkeley/robocasa365-pretrain-atomic \
  --repo-type dataset \
  --local-dir projects/robocasa365/data/atomic

uv pip install --system packaging psutil ninja setuptools wheel
MAX_JOBS=8 uv pip install --system "flash-attn<3" --no-build-isolation
```

</details>

### Eval

<details>
<summary>Evaluation environment and assets</summary>

Use the LeRobot environment above, then run:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git /content/robosuite
cd /content/robosuite
git checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c

cd /content
git clone https://github.com/robocasa/robocasa.git /content/robocasa
cd /content/robocasa
git checkout 4f8a2980def75a55dff96b990745b83540425f09

python -m pip install \
  "numpy==2.2.5" \
  "numba==0.61.2" \
  "scipy==1.15.3" \
  "mujoco==3.3.1" \
  "tianshou==0.4.10" \
  pygame \
  Pillow \
  opencv-python \
  pyyaml \
  pynput \
  tqdm \
  termcolor \
  imageio \
  h5py \
  lxml \
  hidapi \
  gymnasium

python -m pip install -e /content/robosuite
python -m pip install -e /content/robocasa --no-deps

cd /content/robocasa

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets

cd /content/lerobot
```

</details>

## Training budget

GR00T N1.7 used **1× RTX PRO 6000**: **72 hours**, batch 256;
checkpoint **82,500**.

[W&B training log](https://forge.coreweave.com/wandb/siwooyong-seoul-national-university/robocasa_bc/runs/hpy5r4bw)
· [Hugging Face checkpoint](https://huggingface.co/siwooyong021/groot-n17-atomic-checkpoints/tree/main/checkpoints/082500)

## Results

### GR00T N1.7 atomic fine-tuning

Frozen language/vision backbones; trained projector, diffusion action model, and VLLN.

| Model | RoboCasa training data | Atomic-Seen success (%) |
| --- | --- | ---: |
| π0.5 (official reference) | Human300 (Atomic + Composite) | 39.6 |
| GR00T N1.6 (official reference) | Human300 (Atomic + Composite) | 51.1 |
| **GR00T N1.7-3B (this project)** | **Atomic only** | **50.0** |

Reference scores: [official RoboCasa365 leaderboard](https://robocasa.ai/leaderboard).

### SmolVLA memory modeling

Implemented [frame compression and causal temporal memory](../../src/lerobot/policies/smolvla_memory)
for **SmolVLA-0.45B**, using the latest 24 observations.

| Model | Checkpoint (step) | Training loss | Atomic-Seen success (%) |
| --- | ---: | ---: | ---: |
| SmolVLA 0.45B | 200,000 | 0.025009 | 31.11 |
| SmolVLA 0.45B + memory | 80,000 | 0.029234 | 14.26 |

Atomic-only training; **540 evaluation episodes per model**. The memory model may
not yet be fully converged, and its sampling distribution differs from the baseline.
Memory may also reinforce visual shortcuts and reduce rollout robustness.

Expected recovery from earlier mistakes was not observed.
[RoboTTT](https://arxiv.org/html/2607.15275v1#S3.SS3)'s **DAgger Distillation**
uses full rollout history as context and imitation loss only on human corrections,
suggesting that **recovery may need corrective training data alongside memory**.

### Instruction-following failure analysis

Custom instructions probe whether benchmark success extends to instruction
following. All probes start near the refrigerator in **MakeIceLemonade**.

[Full record on Notion, including Xiaomi-Robotics-1](https://app.notion.com/p/2026-10-19-robocasa365-custom-instruct-402cd5137d7680828418f89706cf7448?source=copy_link).

| # | Instruction | Awomo-0.5I observation |
| ---: | --- | --- |
| 1 | Grab the cup on the table and put it inside the microwave. | Pushed cup against closed microwave, dropped it, then grabbed ice container. |
| 2 | Turn around, grab the cup on the table, and throw it as far as possible like a pitcher. | Grasped cup; wandered, then returned it without throwing. |
| 3 | Open all doors and drawers in the kitchen except for the refrigerator doors. | Opened refrigerator; stayed nearby. |
| 4 | Push the toaster on the right table with your arm and knock it onto the floor. | Focused on cup; ignored visible toaster. |
| 5 | Push the cup on the table and knock it onto the floor. | Grasped cup; placed it on another table. |
| 6 | Grab the cup on the table and pour its contents into the sink. | Placed cup at sink without pouring. |
| 7 | Open only one of the refrigerator doors, not both, and take out the lemon inside. | Opened both doors; grasped lemon and wandered. |
| 8 | Do not move to another place, just keep rotating 360 degrees in place. | Opened/closed refrigerator first, then rotated. |
| 9 | Point your gripper upward, then repeatedly close and open it. | Opened/closed refrigerator; twisted arm, then approached refrigerator. |
| 10 | Do not open the refrigerator door; instead, reach toward your own wheels on both sides with your gripper to check how they look. | Opened refrigerator; repeatedly collided with it. |

Related readings behind my hypotheses about these failures:

| Paper | Notes |
| --- | --- |
| [LangForce](https://arxiv.org/abs/2601.15197) | Task-identifying training scenes can make language redundant, enabling visual shortcuts. |
| [When Instructions Retrieve Trajectories](https://arxiv.org/abs/2609.39971) | Policies may respond to language and visual feedback yet select familiar trajectories that do not satisfy the current instruction–scene combination. |

## Evaluation protocol

**18 Atomic-Seen tasks × 30 episodes**, `pretrain` kitchens; macro-average success.
Denoising 16, chunk 16, execute 8, seed 42; 30 async environments/task,
one task at a time, both object registries, official horizons.

## Reproduction

Run from the repository root.

<details>
<summary>Compute statistics and train GR00T</summary>

Compute statistics **before first training**:

```bash
bash projects/robocasa365/scripts/compute_stats.sh projects/robocasa365/data/atomic
bash projects/robocasa365/scripts/train_groot_bc.sh
```

</details>

For fresh training, remove the launcher's resume/config options and choose
a new output directory.

<details>
<summary>Download checkpoint and evaluate</summary>

```bash
hf download siwooyong021/groot-n17-atomic-checkpoints \
  --revision a8854ed93621c84b7a698ea0937ed938d31802a7 \
  --include "checkpoints/082500/pretrained_model/*" \
  --local-dir outputs/groot_n17_robocasa365_atomic_bsz256_steps150k_ck16_gpu1

bash projects/robocasa365/scripts/test.sh
```

</details>

## Code map

| Path | Purpose |
| --- | --- |
| `scripts/train_groot_bc.sh` | Fine-tuning |
| `scripts/test.sh` | Atomic-Seen evaluation |
| `scripts/compute_stats.{py,sh}` | State/action statistics |
| `scripts/train_bc.sh` | Exploratory SmolVLA memory training |
| `src/lerobot/policies/smolvla_memory/` (repository root) | Frame compression and temporal memory |

## References

- [LeRobot](https://github.com/huggingface/lerobot)
- [GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [Xiaomi-Robotics-1](https://arxiv.org/abs/2607.15330)
- [RoboTTT](https://arxiv.org/abs/2607.15275)

## Scope

This project extends LeRobot. Experiment utilities live under `projects/robocasa365`;
the memory policy lives under `src/lerobot/policies/smolvla_memory`.
