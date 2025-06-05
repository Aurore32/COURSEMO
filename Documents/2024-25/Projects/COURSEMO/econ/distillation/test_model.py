import pandas as pd
import numpy as np
import pyperclip

df = pd.read_csv('./econ/distillation/al_econ_combined_dataset_deepseek_r1_distill.csv')
qtypes = list(set(df['Question Type']))
qlist = []
questions = []
markschemes = []
for qtype in qtypes:
    qdf = df[df['Question Type'] == qtype]
    for i in range(36):
        row = qdf.sample(replace=True)
        query = f'''
        Instructions: {row['Instruction'].squeeze()}

        Question: {row['Question'].squeeze()}

        Question Type: {row['Question Type'].squeeze()}
        '''
        qlist.append(query)
        questions.append(row['Question'].squeeze())
        markschemes.append(row['Mark Scheme'].squeeze())

new_df = pd.DataFrame([])
new_df['Query'] = qlist
new_df['Question'] = questions
new_df['Mark Scheme'] = markschemes
new_df.to_csv('./econ/distillation/al-econ-test-queries.csv')