import pandas as pd
import re

dataset = pd.read_csv('./econ/distillation/al_econ_combined_dataset_deepseek_r1_distill.csv')
answers = dataset['Response']

dataset_mcq = pd.read_csv('./econ/distillation/al_econ_multiple_choice_deepseek_r1.csv')
mcq_answers = dataset_mcq['Response']

diagrams = []
count = 0
for answer in answers:
    if 'DIAGRAM:' in answer:
        count += 1
    if len(answer.split()) >= 3:
        diagrams += re.findall(r'(?<=DIAGRAM:)(.*)(?=.)', answer)   
    else: 
       pass
for answer in mcq_answers:
    if 'DIAGRAM:' in answer:
        count += 1
    if len(answer.split()) >= 3:
        diagrams += re.findall(r'(?<=DIAGRAM:)(.*)(?=.)', answer)   
    else: 
       pass
print(len(diagrams))
print(diagrams)

df = pd.DataFrame([])
df['Prompt'] = diagrams
df.to_csv('econ/distillation/al-econ-diagrams-dataset.csv')

print(count)