# GRPO-Lite: A Self-Contained Framework for RLHF Exploration

## Important Note

**This takehome is not designed to be finished.** The goal is not to complete every task—it's to demonstrate your engineering skills and problem-solving approach.

What matters most:
- **Getting models training** for Tasks 1 and 2
- **Thinking logically** through all the issues that arise along the way
- **Documenting your reasoning** and any tradeoffs you encounter

We're evaluating how you approach problems, debug issues, and make engineering decisions—not whether you tick every box.

## 1. Overview

Welcome to your interview! This project provides a lightweight, self-contained framework for training language models using a simplified version of GRPO. The goal is to give you a hands-on opportunity to work with modern RLHF techniques, analyse model behaviour, and even design your own reward systems.

The codebase is written in PyTorch and intentionally avoids heavy dependencies like Hugging Face's TRL to ensure the core logic is transparent and accessible. You will be working directly with the training loop, loss implementation, and evaluation pipeline.

The interview is structured into a series of tasks that progress from core implementation to open-ended analysis and creative extension.

Good luck!

## 2. Setup

**Cloud Compute Setup (Vast.ai):**
Charlie will get you set up with an account on [Vast.ai](https://vast.ai/) for GPU resources.
-   For training the smaller `Qwen2-0.5B` model, renting a single **H100** GPU is sufficient and cost-effective.
-   For the scaling laws analysis in Task 3 involving larger Qwen models, you will likely need a node with **2-4 x H100** GPUs. Part of the task is managing the computational resources efficiently.

## 3. Codebase Structure

The project is organized into several key files:

-   `main.py`: The main entry point for training and evaluation. It contains the primary training loop, calls the evaluator, and is where you will implement the loss function.
-   `llms.py`: A simple utility for loading language models and tokenizers from Hugging Face.
-   `evaluator.py`: Defines the reward evaluation logic. It includes the `RewardEvaluator` abstract base class and a concrete `GSM8kEvaluator` for the math reasoning task.
-   `rl_datasets.py`: Manages data loading. Currently configured for the GSM8K dataset.
-   `plotter.py`: A script to generate plots from training and evaluation logs, helping you visualize results.
-   `utils.py`: Contains miscellaneous helper functions for seeding, logging, and calculating log probabilities.

## 4. Interview Tasks

Please complete the tasks in the order presented.

---

### Task 1: (Engineering) Environment Setup & Bug Fixes

Your first task is to get the development environment running and fix some bugs in the codebase.

#### Part A: Environment Setup with uv

The repository currently uses a `requirements.txt` file, but we'd like you to convert it to use [uv](https://github.com/astral-sh/uv) for dependency management.

**Objectives:**
1.  Convert the project to use `uv` with a proper `pyproject.toml`
2.  Get Flash Attention 2 building and working—this is required for efficient training. The codebase expects Flash Attention to be available.

This is an engineering test: we want to see how you handle dependency management and build issues on a real ML project.

#### Part B: Fix the Parser and Reward Function

There are bugs in the response parsing and reward computation logic. The evaluator expects completions in a specific format (`<reasoning>...</reasoning><answer>...</answer>`), but the parsing and reward assignment have issues.

**Objectives:**
1.  Review `evaluator.py` and identify the bugs in parsing and reward computation
2.  Fix the issues so that correct answers receive positive rewards and incorrect answers receive appropriate penalties

---

### Task 2: (Core) Implement the Loss Function & Train a Model

Now implement the core GRPO loss function and train a model.

#### Part A: Implement the `compute_loss` Function

Navigate to the `compute_loss` function in `main.py`. It is currently empty.

**Objective:**
Implement a per-token, KL-regularized policy gradient loss. The objective is to maximize rewards while penalizing KL divergence from the reference policy on a per-token basis.

**Key Inputs:**
-   `model`: The policy model being trained.
-   `base_model`: The reference model (a frozen copy of the initial model).
-   `prompt_completion_ids`: The full token sequences.
-   `completion_ids`: The tokens of the generated completions.
-   `attention_mask`: Attention mask for the full sequence.
-   `completion_mask`: A mask to ignore padding tokens in the completions.
-   `advantages`: The calculated advantages for each completion, which represent the normalized rewards.

**Implementation Steps:**

1.  **Calculate Per-Token Log Probabilities:**
    -   Use the `utils.get_per_token_logps` function to get the log probabilities of the `completion_ids` under the `model`.
    -   In `torch.inference_mode()`, do the same for the `base_model` to get the reference log probabilities (`ref_per_token_logps`).

2.  **Calculate Per-Token KL Divergence:**
    -   Compute the forward KL divergence between the reference and policy distributions for each token. The formula is:
        `kl = exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1`

3.  **Calculate the Per-Token Policy Objective:**
    -   The core of the policy gradient update uses an importance sampling ratio.
    -   Multiply this ratio by the `advantages` to scale the updates based on reward.

4.  **Combine into Final Per-Token Loss:**
    -   Combine the policy objective with the KL penalty.

5.  **Aggregate the Loss:**
    -   Apply the `completion_mask` to the `per_token_loss` to exclude padding tokens from the loss calculation.
    -   Sum the masked losses along the sequence dimension and normalize by the number of completion tokens for each sequence.
    -   Finally, take the mean across the batch to get the final scalar `loss`.

6.  **Return Metrics:**
    -   Return the `loss`.
    -   Also compute and return a dictionary of metrics, including the mean KL divergence (`kl`) and the average `response_length`.

#### Part B: Train and Analyze Results

Once your loss function is implemented, train a small model on the GSM8K dataset.

**Instructions:**

1.  **Run Training:** Execute the main script to start training. We recommend starting with the `Qwen/Qwen2-0.5B-Instruct` model, which is fast to train.
    ```bash
    python main.py \
        --model_name "Qwen/Qwen2-0.5B-Instruct" \
        --num_train_iters 1000 \
        --eval_iterations 100 \
        --output_dir "output/Qwen-0.5B"
    ```
    Feel free to adjust the training arguments in `main.py` as you see fit.

2.  **Monitor Output:** Training logs, evaluation results, and model checkpoints will be saved in the specified `output_dir`.

3.  **Visualize Metrics:** Use the `plotter.py` script to visualize the training dynamics (also results saving and plotting can definitely be cleaned up a lot).
    ```bash
    python plotter.py --log_dir "output/Qwen-0.5B"
    ```
    This will generate a `training_plots.pdf` in the log directory. Review the plots for loss, rewards, KL divergence, and evaluation accuracy. Does the model appear to be learning successfully?

---

### Task 3: (Analysis) Scaling Laws

Now, conduct a small-scale analysis of how model size affects performance on the GSM8K task.

**Objective:**
Explore the relationship between model scale, performance, and data efficiency.

**Instructions:**

1.  **Train Multiple Models:** Train at least two different model sizes from the same family (e.g., `Qwen2-0.5B` and `Qwen2-1.5B`). If you have the computational resources, you can add a larger model like `Qwen2-7B`. Train them for the same number of iterations.
2.  **Plot Performance vs. Scale:** Create a plot of final evaluation accuracy vs. model size (number of parameters).
3.  **Analyze Data Efficiency:** Plot the evaluation accuracy over training steps for each model on the same graph. Does the larger model learn faster? Does it achieve a higher peak performance?
4.  **Summarize Findings:** Write a brief summary of your conclusions. What are the trade-offs you observed? You can create a new markdown file or a Jupyter notebook for this analysis.

---

### Task 4: (Analysis) RL vs. SFT Efficiency

Compare the effectiveness and data efficiency of our RL approach (GRPO) against standard Supervised Fine-Tuning (SFT).

**Objective:**
Investigate how RL training compares to fine-tuning on a small, high-quality dataset.

**Instructions:**

1.  **Generate a "Gold Standard" Dataset:**
    -   Write a script to generate high-quality solutions for ~500-1000 problems from the GSM8K training set.
    -   Use a powerful LLM for generation (e.g., via an API for GPT-4, or a large local model like Mixtral or Llama-3-70B).
    -   Ensure the generated solutions strictly follow the required `<reasoning>...</reasoning><answer>...</answer>` format. Save these as your SFT dataset.

2.  **Implement and Run SFT:**
    -   Write a new, simple script to perform SFT on a base model (e.g., `Qwen2-0.5B-Instruct`) using your gold-standard dataset. You can use Hugging Face's `Trainer` API for this.
    -   Train the model and evaluate its accuracy on the GSM8K test set.

3.  **Compare and Analyze:**
    -   How does the SFT model's performance compare to the GRPO-trained model from Task 2?
    -   To analyze data efficiency, train several SFT models on subsets of your gold data (e.g., 100, 250, 500 examples). Plot the SFT accuracy vs. the number of training examples.
    -   Compare this plot to the accuracy-over-time plot from your GRPO run. Roughly how many GRPO training steps appear to be as effective as one high-quality SFT example?
    -   Write up your analysis and conclusions.

---

### Task 5: (Extension) LLM-as-a-Judge for a Creative Task

If you have time, extend the framework to a non-verifiable, creative domain.

**Objective:**
Implement an LLM-as-a-judge reward model and use it to train a policy on a creative writing task.

**Instructions:**

1.  **Define a Task and Dataset:**
    -   Choose a simple creative task, for example, "Given an opening line, write a short, one-paragraph story."
    -   Create a small dataset of prompts (e.g., 50-100 opening lines). You will need to create a new `DataLoader` in `rl_datasets.py`.

2.  **Implement an LLM-as-a-Judge Evaluator:**
    -   Create a new class, `CreativeWritingEvaluator`, in `evaluator.py` that inherits from `RewardEvaluator`.
    -   In its `compute_rewards` method, call an external LLM (this can be the same model you are training, another local model, or an API call) to act as a "judge".
    -   Design a prompt for the judge model that asks it to rate the generated story on a numeric scale (e.g., 1 to 5) based on criteria like creativity, coherence, and engagement. You'll need to be careful here - a lot of judges don't have good discriminatory power, and you'll need some responses to be judged as good and some as poor, otherwise GRPO won't work. 
    -   The reward for each completion will be the score returned by the judge LLM. Be sure to handle parsing the judge's output robustly.

3.  **Train the Model:**
    -   Run `main.py` using your new creative dataset and LLM-as-a-judge evaluator.
    -   Examine the generated stories. Does the model's writing quality improve over time? What are the challenges of using an LLM as a reward source?
