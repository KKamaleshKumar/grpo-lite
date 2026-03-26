import json
import itertools
import sys

from rl_datasets import build_gsm8k_dataloaders
from openai import OpenAI

client = OpenAI(api_key="enter your api key here")


trainloader, _ = build_gsm8k_dataloaders()


SYSTEM_PROMPT = trainloader.pre_prompt

def generate_solution(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

sft_data = []
for i, (question, answer) in enumerate(itertools.islice(trainloader, 700)):
    solution = generate_solution(question)
    sft_data.append({
        "question": question,
        "solution": solution,
        "answer": answer
    })
    if i % 50 == 0:
        print(f"Generated {i}/1000")

        with open("sft_data.json", "w") as f:
            json.dump(sft_data, f, indent=2)

with open("sft_data.json", "w") as f:
    json.dump(sft_data, f, indent=2)
