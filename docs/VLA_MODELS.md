# VLA models: a learning guide

*Written 2026-10-31, for choosing a policy for this project. Numbers marked ⚠️ come from memory or secondary sources and were not checked against the official paper or repo. Sources are at the end.*

## The basics

A **vision-language-action model (VLA)** takes camera images, the robot's joint state and a text instruction, and outputs robot actions. Most VLAs are built the same way:

1. **A vision-language model (VLM)** that was pretrained on internet images and text. It turns the images and the instruction into features.
2. **An action head** (often called the *action expert*) that turns those features into motor commands.
3. **Robot pretraining** on many robot demonstrations, so the model already knows roughly how arms move.
4. **Fine-tuning** on your own robot's demonstrations. This is the step we run.

Terms used below:

| Term | Meaning |
|---|---|
| **Action chunk** | The model predicts many future actions at once (for example 50) instead of one. This gives smoother motion, but the robot runs part of the chunk without looking again |
| **Flow matching / diffusion** | The action head starts from random noise and refines it into actions over a few steps. Good for smooth, continuous motion |
| **Action tokens** | Actions are turned into discrete "words" and generated one at a time, like text. Simpler, but slower and less smooth |
| **Frozen vision** | Only the action head is trained; the vision part keeps its pretrained weights. Saves memory, but the model can't learn to see new objects better |
| **LoRA** | A cheap fine-tuning method that trains small add-on weight matrices instead of the whole model |
| **VRAM** | GPU memory. **Our laptop has 8 GB**, which rules out fine-tuning most large models locally |

## Comparison at a glance

| Model | Size | Language? | Action output | Fine-tune memory | Licence | Fits our 8 GB? |
|---|---|---|---|---|---|---|
| **SmolVLA** | 0.45 B | ✅ | Flow matching, chunk 50 | ~5–6 GB at batch 12 (measured, vision frozen) | Apache-2.0 ⚠️ | ✅ **What we use** |
| **X-VLA** | 0.9 B | ✅ | Flow matching | Unknown; only ~9 M parameters need tuning | Apache-2.0 | ⚠️ Maybe, untested |
| **π0** | ~3.3 B ⚠️ | ✅ | Flow matching, chunk 50 | LoRA > 22.5 GB, full > 70 GB | Apache-2.0 (+ Gemma terms) | ❌ Inference only (> 8 GB) |
| **π0-FAST** | ~3 B ⚠️ | ✅ | Action tokens | Like π0 | Apache-2.0 (+ Gemma terms) | ❌ |
| **π0.5** | ~3 B ⚠️ | ✅ | Flow matching | Like π0 | Apache-2.0 (+ Gemma terms) | ❌ |
| **GR00T N1.5** | 3 B | ✅ | Flow matching, chunks | No official figure; N1.7 needs ~35–40 GB ⚠️ | Code Apache-2.0; **weights non-commercial** (NVIDIA licence) | ❌ |
| **OpenVLA** | 7 B | ✅ | Action tokens, 1 step | LoRA needs A100-class GPUs | MIT | ❌ |
| **OpenVLA-OFT** | 7 B | ✅ | Continuous, chunked | ~27–35 GB per GPU ⚠️ | MIT | ❌ |
| **ACT** *(not a VLA)* | ~80 M ⚠️ | ❌ | Chunked (CVAE) | Small | MIT ⚠️ | ✅ |
| **Diffusion Policy** *(not a VLA)* | ~0.1–0.3 B ⚠️ | ❌ | Diffusion, chunked | Small | MIT ⚠️ | ✅ |

## The models

### SmolVLA (Hugging Face, 2025): our current model

**Specifications** (checked on our checkpoint):
- 450 M parameters in total, 100 M trainable when the vision part is frozen.
- **Vision-language part:** SmolVLM2-500M, cut to 16 layers.
- **Action head:** flow matching, 10 refinement steps, chunk of 50 actions. State and action are padded to 32 numbers.
- **Robot pretraining:** community LeRobot datasets, mostly low-cost SO-100 arms ⚠️.

**Advantages:**
- Fits a laptop GPU: we train at batch 12 in about 5.8 GB.
- Built into LeRobot, so dataset → training → checkpoint works with one command.
- Fast enough to run live (about 110 ms per chunk on our GPU).
- Learns grasp motions well. Our near-misses reach 3 cm.

**Disadvantages:**
- **Weak at picking *the named* object among several.** A 2026 study found grasping skill and instruction following "decoupled" in SmolVLA and π0.5. Our own result matches: arm choice was a 50/50 coin flip.
- Small vision part, usually frozen, so it can't learn to tell look-alike objects apart (our instruments are ~25 px grey shapes).
- Pretrained mostly on single, cheap arms. Nothing like a bimanual FR3.

### X-VLA (2025, ICLR 2026)

**Specifications:**
- 0.9 B parameters.
- **Vision-language part:** Florence-2 encoder.
- **Action head:** a flow-matching transformer.
- **Soft prompts:** each robot gets a small set of learned "soft prompts", embeddings that tell the model which robot it is controlling.
- **Robot pretraining:** 290 k episodes from 7 robot platforms.
- **In LeRobot:** yes.

**Advantages:**
- **Cheap to adapt to a new robot.** The paper tunes only ~1 % of the weights (9 M) for a new robot and reports 93 % on the LIBERO benchmark.
- Designed for different robots ("cross-embodiment") from the start.
- The smallest of the large models; plausibly trainable on 8 GB with the backbone frozen. Untested.

**Disadvantages:**
- Newer, with a smaller community and fewer worked examples than SmolVLA or π0.
- No published memory figure for fine-tuning; we'd have to measure.
- Unknown whether it handles "which object" better than SmolVLA.

### π0, π0-FAST, π0.5 (Physical Intelligence, 2024–2025)

**Specifications:**
- **Vision-language part:** PaliGemma (~3 B) ⚠️.
- **Action heads:**
  - π0: a ~300 M flow-matching action expert, chunk of 50 ⚠️.
  - π0-FAST: action tokens instead.
  - π0.5: adds training on web data for better generalisation.
- **Code:** in `openpi`, JAX and (since Sep 2025) PyTorch.

**Advantages:**
- Among the strongest open VLAs; large, varied robot pretraining, including two-arm robots.
- π0.5 is specifically aimed at working in new scenes and with new objects.
- Large community; LeRobot integration exists.

**Disadvantages:**
- **Memory:** inference needs > 8 GB; LoRA fine-tuning > 22.5 GB; full fine-tuning > 70 GB. Not possible on our laptop.
- The PyTorch version doesn't support LoRA yet, so cheap fine-tuning needs the JAX version.
- π0.5 is also named in the 2026 study above as having the same object-grounding weakness.

### GR00T N1.5 (NVIDIA, 2025)

**Specifications:**
- 3 B parameters.
- **Vision-language part:** a pretrained one, NVIDIA's Eagle ⚠️.
- **Action head:** a flow-matching action transformer that outputs chunks.
- **In LeRobot:** yes (`policy.type=groot`), but it needs Flash Attention and a CUDA GPU. Newer LeRobot versions may only support N1.7 ⚠️.

**Advantages:**
- Built for humanoids and two-arm robots.
- Fits the NVIDIA tool chain (Isaac Sim / Isaac Lab).
- LeRobot's version matches NVIDIA's reported LIBERO score (87 %).

**Disadvantages:**
- The weights are under an NVIDIA **non-commercial** licence (the code is Apache-2.0).
- Out of the box it only works on robots it was trained for. The clearest case is the single-arm Franka "DROID" setup: the `GR00T-N1.6-DROID` checkpoint runs a pick-and-place task in Isaac Lab Arena with no fine-tuning. Any other robot, ours included, needs fine-tuning.
- No official memory figure for N1.5; the newer N1.7 recommends 40 GB+ for fine-tuning ⚠️. Not possible on our laptop.
- Version churn between N1.5, N1.6 and N1.7 and LeRobot releases.

### OpenVLA and OpenVLA-OFT (Stanford and others, 2024–2025)

**Specifications:**
- 7 B parameters, built on Llama 2 with DINOv2 + SigLIP vision ⚠️.
- **Action output:**
  - OpenVLA: one action at a time, as tokens.
  - OFT ("optimized fine-tuning"): continuous, chunked actions, decoded in parallel.

**Advantages:**
- Well studied and documented; MIT licence.
- OFT fixes most of the original's speed and smoothness problems.

**Disadvantages:**
- Too large for us: the official LoRA recipe used 8 × A100 80 GB; community figures put OFT at ~27–35 GB per GPU ⚠️.
- The original outputs one step at a time, so motion is slow and jerky compared with chunked models.

### Not VLAs, but worth knowing: ACT and Diffusion Policy

Both are **imitation-learning policies without language**: images + state in, actions out.

**ACT** (Action Chunking with Transformers, from the ALOHA project):
- **Advantages:** small (~80 M ⚠️); trains from scratch on ~50 demos in under an hour; very precise for one fixed task.
- **Disadvantage:** can't be told *which* object to pick. We dropped it in v1 for that reason.

**Diffusion Policy:**
- **Advantages:** very smooth and robust motion; handles tasks with several valid solutions.
- **Disadvantage:** also has no language.

**Where they fit:** both are reasonable choices for **Plan B's final grasp-and-place step**, where the planner has already chosen the target and no language is needed.

## What this means for this project

| Question | Answer |
|---|---|
| Can we use a bigger model locally? | **No.** Everything above ~1 B needs ≥ 22 GB to fine-tune. Only SmolVLA, X-VLA (maybe), ACT and Diffusion Policy fit 8 GB |
| Would a bigger model fix "which instrument"? | **Not guaranteed.** The 2026 study found the same weakness in π0.5 |
| Best next move if v2.5 fails | **Plan B:** a detector + planner choose and reach the target; a small policy (SmolVLA, or ACT) only does the close-up grasp and place |
| Worth trying later | **X-VLA:** it's small enough to test on 8 GB and built for new robots |

## Sources

- SmolVLA checkpoint specs: measured on `outputs/smolvla_instruments_v1` (parameter counts, config).
- [openpi repository](https://github.com/Physical-Intelligence/openpi): π0 / π0-FAST / π0.5 memory, licence, PyTorch status.
- [GR00T N1.5 model card](https://huggingface.co/nvidia/GR00T-N1.5-3B): weights licence. [Isaac Lab Arena GR00T quickstart](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/running_a_real_policy/gr00t.html): out-of-the-box DROID demo.
- [LeRobot GR00T docs](https://huggingface.co/docs/lerobot/en/groot): GR00T N1.5 size, LeRobot integration, LIBERO score. Memory figures from [Isaac-GR00T N1.7](https://github.com/precognitionlab/Isaac-GR00T-N1.7) and [Seeed course](https://wiki.seeedstudio.com/robotics/soarm_physical_ai_course_chapter_21/) ⚠️.
- [X-VLA paper](https://arxiv.org/html/2510.10274v1), [X-VLA repo](https://github.com/2toinf/X-VLA), [model card](https://huggingface.co/2toINF/X-VLA-WidowX/blob/main/README.md).
- OpenVLA: [LIBERO LoRA model card](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial/commit/962318cec55ac10993ff0f5f43eda9a270b4c873); OFT memory from a [community page](https://playbooks.com/skills/orchestra-research/ai-research-skills/openvla-oft) ⚠️.
- Object-grounding weakness: [*Robust Skills, Brittle Grounding*](https://arxiv.org/abs/2602.24143) (abstract checked; the numbers weren't), [OBEYED-VLA](https://arxiv.org/pdf/2512.22519).
- Community SmolVLA fine-tunes: [full fine-tune card](https://huggingface.co/Mehul2112/smolvla_black_cube_fullft).
