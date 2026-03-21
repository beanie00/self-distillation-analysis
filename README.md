<div align="center">

# Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?

[![Paper](https://img.shields.io/badge/arXiv-xxxx.xxxx-F46565?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxx) [![Code](https://img.shields.io/badge/GitHub-Code-000000?style=flat&logo=github&logoColor=white)](https://github.com/lasgroup/SDPO) [![W&B](https://img.shields.io/badge/W%26B-Logs-06B6D4?style=flat&logo=weightsandbiases&logoColor=white)](https://wandb.ai/jonhue/SDPO?nw=mgotcx6kk7) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-FEE47D?style=flat)](https://huggingface.co/your-repo-here)

</div>

## Introduction

<div align="center">
  <video src="figures/intro.mp4" width="800" controls>
    Your browser does not support the video tag.
  </video>
</div>

## Installation

### Data Preparation
We primarily use the [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset for training. You can download it by running:
```bash
bash experiments/math/prepare_dapo_data.sh
```

The dataset for Section 6.2 (*"Relationship Between Task Coverage and Learning Performance"*) and the evaluation datasets are available in the `/data/math` directory.

### Prompt Reformatting
The DAPO-Math-17k dataset prompts the model to produce answers in the `Answer:` format. The following script reformats these prompts to elicit `\boxed{}` answers instead. We found that this reformatting consistently improved the model's accuracy.
```bash
python experiments/math/change_math_prompts.py
```

<details>
<summary>See before/after example</summary>

**Before:**
```python
[{'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after "Answer:".', 'role': 'user'}]
```
**After:**
```python
[{'content': 'In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\nPlease reason step by step, and put your final answer within \\boxed{}.', 'role': 'user'}]
```

</details>


### Configuration

Below are the key hyperparameter differences from the default [SDPO](https://github.com/lasgroup/SDPO) settings:

| Parameter | Value |
|-----------|-------|
| `train_batch_size` | 256 |
| `ppo_mini_batch_size` | 128 or 64 (minimal difference) |
| `max_prompt_length` | 2048 |
| `max_response_length` | 20480 |
| `max_reprompt_len` | 22528 |
| `teacher_update_rate` | 0 (main) / 0.05 (ablation) |
| `remove_thinking_from_demonstration` | `False` for SDPO ($c=s$) / `True` for SDPO ($c = s_{\setminus\text{think}}$) |

Note: The original `remove_thinking_from_demonstration` implementation strips `<think>...</think>` from the student response. However, for models like DeepSeek-R1-Distill-7B, the chat template already includes an opening `<think>` tag in the prompt, so the student response begins without an opening `<think>` but still contains a closing `</think>`. We modified `_remove_thinking_trace` in `verl/trainer/ppo/ray_trainer.py` to handle this case.

## Run Examples

### Analyzing Reasoning Behavior Under Richer Information (Section 3)
```bash
cd analyzing_reasoning_behavior

# Collect 100 questions from DAPO-Math-17k where the base model solves at a rate between 0.125 and 0.5, along with their solutions
bash eval_dapo_dataset.sh

# Re-evaluate the model with solutions provided as hints
bash eval_with_hint.sh

# Generate a summary report
bash make_report.sh
```

The collected results are then used to construct the dataset for the SFT experiments in Section 4:
```bash
bash make_sft_dataset.sh
```

### GRPO and SDPO Training
To train with GRPO or SDPO on the full dataset, run:
```bash
bash experiments/math/run_math_grpo.sh
bash experiments/math/run_math_sdpo.sh
```

For reproducing Section 6.2 (*"Relationship Between Task Coverage and Learning Performance"*) with a reduced question set, run:
```bash
bash experiments/math/run_math_grpo_small_question.sh
bash experiments/math/run_math_sdpo_small_question.sh
```

### Evaluation
After training, merge the saved FSDP checkpoints into HF format before running evaluation.
```bash
cd eval
bash merge_models.sh
bash eval.sh
```

## Citation
If you find this work helpful, please consider citing:

## Attribution
