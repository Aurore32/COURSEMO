import requests
import json
import pandas as pd


url1 = "https://al-econ-coursemo-36-1ae8570-v1.app.beam.cloud"
headers = {
  "Authorization": "Bearer L2gFNLRXiWQpj21-k4aiuIMwtOM7BR_Qvu11-bKckxGXrh4SqStZz-AimSwkOVEJiPg6eTrBiVxfQ3_UQeKRLw==",
  "Content-Type": "application/json"
}

response2 = requests.request("POST", url1,
headers=headers,
    )
