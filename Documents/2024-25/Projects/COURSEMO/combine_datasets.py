import pandas as pd
dataset1 = pd.read_csv('al_econ_data_response_deepseek_r1_distill.csv')
dataset2 = pd.read_csv('al_econ_data_response_deepseek_r1_distill_v2.csv')
df = pd.concat([dataset1, dataset2])
print(df)
df = df.reset_index()
df = df.drop(columns='Unna')
print(df)
df.to_csv('al_econ_data_response_deepseek_r1_distill_combined.csv') 