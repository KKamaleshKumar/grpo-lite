import json
import os
import re
import torch
import argparse
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from tqdm import tqdm

import llms
from rl_datasets import build_gsm8k_dataloaders


def prepare_dataset(data, num_examples, tokenizer, pre_prompt, max_length):
    subset = data[:num_examples]
    texts = []
    for ex in subset:
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": ex["question"]}
        ], tokenize=False, add_generation_prompt=True)
        full_text = prompt + ex["solution"] + tokenizer.eos_token
        texts.append({"text": full_text, "prompt_len": len(tokenizer(prompt)["input_ids"])})

    def tokenize_fn(ex):
        out = tokenizer(ex["text"], truncation=True, max_length=max_length, padding="max_length")
        labels = out["input_ids"].copy()
        for i in range(ex["prompt_len"]):
            labels[i] = -100
        for i, tok in enumerate(out["input_ids"]):
            if tok == tokenizer.pad_token_id:
                labels[i] = -100
        out["labels"] = labels
        return out

    dataset = Dataset.from_list(texts).map(tokenize_fn)
    return dataset.remove_columns(["text", "prompt_len"])


def evaluate(model, tokenizer, pre_prompt, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for question, answer in tqdm(testloader, desc="Evaluating"):
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
            predicted = re.findall(r'-?\d+\.?\d*', match.group(1))[-1] if match else None
            if predicted and predicted.strip() == answer.strip():
                correct += 1
            total += 1
    model.train()
    return correct / total * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="sft_data.json")
    parser.add_argument("--output_dir", type=str, default="output/sft_experiments")
    parser.add_argument("--subset_sizes", type=int, nargs="+", default=[100, 250, 500,700])
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainloader, testloader = build_gsm8k_dataloaders()
    pre_prompt = trainloader.pre_prompt

    with open(args.data_path) as f:
        data = json.load(f)

    results = {}
    for n in args.subset_sizes:

        model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)

        dataset = prepare_dataset(data, n, tokenizer, pre_prompt, args.max_length)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(args.output_dir, f"sft-{n}"),
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-5,
                bf16=True,
                logging_steps=10,
                save_steps=0,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                report_to="none"
            ),
            train_dataset=dataset,
        )
        trainer.train()

        testloader.reset()
        accuracy = evaluate(model, tokenizer, pre_prompt, testloader, device)
        results[n] = accuracy


        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)


    sizes = sorted(results.keys())
    accs = [results[k] for k in sizes]
    plt.plot(sizes, accs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of SFT Examples")
    plt.ylabel("Evaluation Accuracy (%)")
    plt.title("SFT Dataset Size vs. Evaluation Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "sft_results.pdf"))