import pandas as pd

print(list(pd.read_csv('igcse-physics-{}.csv'.format('electromagnetism')).columns)[1:])