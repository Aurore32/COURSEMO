import requests
import json

url = "https://al-econ-coursemo-38-b1516fd-v1.app.beam.cloud"
headers = {
  "Authorization": "Bearer L2gFNLRXiWQpj21-k4aiuIMwtOM7BR_Qvu11-bKckxGXrh4SqStZz-AimSwkOVEJiPg6eTrBiVxfQ3_UQeKRLw==",
  "Content-Type": "application/json"
}


payload = {'prompt':f"""
      Who won the 2020 presidential election, and what were its consequences?
  """}

with requests.get(url, headers=headers, data=json.dumps(payload), stream=True) as r:  
    for chunk in r.iter_content(1024):  
        print(chunk.decode('utf_8'))