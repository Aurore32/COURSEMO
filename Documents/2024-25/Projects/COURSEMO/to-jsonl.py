import json
import pandas as pd

train_data = pd.read_csv('al_econ_train.csv')
instructions = train_data['Instruction']
questions = train_data['Question']
answers = train_data['Answer']

with open('al_econ_train.jsonl', 'w') as output:
    for instruction, question, answer in zip(instructions, questions, answers):
        entry = {"messages": [{"role": "system", "content": instruction}, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]}
        print(json.dumps(entry), file=output)

