import requests
import json
import pandas as pd

url = "https://al-econ-coursemo-30-c8a6c0c-v1.app.beam.cloud"
headers = {
  "Authorization": "Bearer L2gFNLRXiWQpj21-k4aiuIMwtOM7BR_Qvu11-bKckxGXrh4SqStZz-AimSwkOVEJiPg6eTrBiVxfQ3_UQeKRLw==",
  "Content-Type": "application/json"
}

df = pd.read_csv('./econ/distillation/al-econ-test-queries.csv')

queries = df['Query'].tolist()[363:]
questions = df['Question'].tolist()[363:]
markschemes = df['Mark Scheme'].tolist()[363:]


for i in range(len(queries)):
  old_df = pd.read_csv('./econ/al_econ_full_test_trained.csv')
  query = queries[i]
  question = questions[i]
  markscheme = markschemes[i]
  payload = {'prompt':f"""
      {query}
  """,
  'isimage':False}

  with requests.get(url, headers=headers, data=json.dumps(payload), stream=True) as r:  
    for chunk in r.iter_content(1024):  
        print(chunk)

'''  new_df = pd.DataFrame([])
  new_df['Question'] = [question]
  new_df['Mark Scheme'] = [markscheme]
  new_df['Response'] = [response2['text']]
  newest_df = pd.concat([old_df, new_df], ignore_index=True)
  newest_df.drop(newest_df.columns[newest_df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
  newest_df.to_csv('./econ/al_econ_full_test_trained.csv')'''