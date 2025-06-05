import pandas as pd

df = pd.read_csv('./econ/structured/al_econ_structured_dataset.csv')
df = df[df['Question Type'] == 'A-Level, [13] marks']

df.to_csv('./econ/structured/al_econ_structured_dataset_13_mark.csv')