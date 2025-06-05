from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset

# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model

import torch
import evaluate

import numpy as np
import pandas as pd

# tqdm library makes the loops show a smart progress meter.
from unsloth import to_sharegpt
from datasets import load_dataset
from unsloth import standardize_sharegpt
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from tqdm import tqdm
tqdm.pandas()

dataset = load_dataset('csv', data_files='al_econ_data_response_deepseek_r1_distill_combined.csv', split='train').select_columns(['Extract', 'Question', 'Mark Scheme', 'Response'])
dataset = standardize_sharegpt(dataset)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "...", # Change later
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = 4096,
)

tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(examples):
    convos = []
    # Iterate through each item in the batch (examples are structured as lists of values)
    for question, questiontype, markscheme in zip(examples['Question'], examples['Type'], examples['Mark Scheme']):
        tool_user = {"content": """
    You are an A-Level economics expert with full knowledge of the A-Level syllabus. You will be given an A-Level or AS-Level economcis structured essay question, the number of marks of the question, as well as the mark scheme to the question, which should be referenced to generate a response that would obtain full marks according to it. 
    Generate responses in continuous prose (no headings, bullets, or markdown). Follow these rules strictly:

    Formatting & Structure
    1. Never use bullets, lists, or headings—write in plain paragraphs.
    2. Stick to economic theory - do not reference real-world examples, even if given in the mark scheme. Your answer should be something a top-scoring student would write in an actual exam.
    3. You will be given the number of marks as well as the type of the question (AS-Level or A-Level). Answer the question based on its number of marks according to these guidelines:
        - AS-Level, [8] marks: 400 words. 300 words explanation and analysis of question. 100 word evaluation of question and conclusion.
        - AS-Level, [12] marks: 700 words. 500 words for two paragraphs, one supporting the statement in the question and one opposing it. 200 words for an evaluation of whether the question is true or not and a conclusion.
        - A-Level, [12] marks: 700 words. 600 to 700 words for an explanation and analysis of the question. 100 words for an evaluation of the question if mentioned in the mark scheme.
        - A-Level, [13] marks: 750 words. 600 to 700 words for an explanation and analysis of the question. 100 words for an evaluation of the question if mentioned in the mark scheme.
        - A-Level, [20] marks: 1000 words. 750 words for an explanation and analysis of the question. 250 words for an evaluation of the question plus a conclusion.
        - A-Level, [25] marks: 1200 words. 900 words for an explanation and analysis of the question. 300 words for an evaluation of the question plus a conclusion.
    - The content of your answer should be based on the mark scheme. Elaborate with clear, detailed chains of economic analysis on each point in the mark scheme.
    - The mark scheme may also mention level descriptors ("Level 4", "Level 4", etc.) The highest level (e.g. Level 4) will be the one for top marks. Please make sure you have followed the level descriptors for the highest level.

    Diagrams
    - If your answer needs to reference a diagram:
        - Indicate this via the text (DIAGRAM: (description of what the diagram is showing)).
        - Reference and explain key elements (e.g., "As shown in the monopoly diagram, profit maximization at MR=MC leads to a deadweight loss triangle ABC").
        - Link to analysis (e.g., "The shift from S1 to S2 in the supply diagram illustrates how subsidies reduce production costs").

    Non-Negotiable
    - DO NOT generate any headings, bold text or formatting. Generate your text in plain prose.
    - DO NOT include any real-world examples in your answer. Economic analysis only.          
    - Your explanation paragraphs combined should at least be two times longer than your evaluation, if the question asks for it. Expand on the analysis points from the mark scheme more than on evaluation.
    """, "role": "system"}
    
        ques_user = {"content": f"""
        Question: {question}

        Question Type: {questiontype}

        Mark Scheme: {markscheme}

        Remember to answer in continuous prose and follow the guideline for this question based on its number of marks, as provided in the system prompt.
        """,
            "role": "user"
        }

        convos.append([tool_user, ques_user])

    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"query": texts}

# Apply the formatting on dataset
dataset = dataset.map(formatting_prompts_func, batched = True)
print(dataset[0])

dataset = 1 # Fill in later

ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)
ppo_trainer = PPOTrainer(model=model, config=ppo_config, train_dataset=dataset, tokenizer=tokenizer)

def reward_function(query, response):
    pass

for epoch in range(1):
    for batch in ppo_trainer.dataloader:
        queries = batch['query']
        response_tensors = ppo_trainer.generate(
            queries,
            max_new_tokens = 2048,
            temperature = 0.6,
            top_p = 0.9,
            pad_token_id = tokenizer.eos_token_id,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        rewards = [reward_function(q, r) for q, r in zip(queries, batch["response"])]
        rewards = torch.tensor(rewards).to(model.device)
        
        # PPO Update
        stats = ppo_trainer.step(
            response_tensors,
            rewards,
            response_masks = [torch.ones_like(r) for r in response_tensors],
        )
        
        # Log metrics
        print(f"Epoch {epoch} | Mean Reward: {rewards.mean().item():.2f}")