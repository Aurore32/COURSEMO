import requests
import json
import base64
from PIL import Image

url = "https://coursemo-ready-63a669e-v4.app.beam.cloud"
headers = {
  "Authorization": "Bearer L2gFNLRXiWQpj21-k4aiuIMwtOM7BR_Qvu11-bKckxGXrh4SqStZz-AimSwkOVEJiPg6eTrBiVxfQ3_UQeKRLw==",   
  "Content-Type": "application/json"
}     

with open('./deployment/image5.jpg', "rb") as image_file:         
    imagequery = base64.b64encode(image_file.read()).decode('utf-8')
    payload = {'textprompt': 'Define the concept of price elasticity of demand and explain its formula',
    'imageprompt': '',
    'type': '',
    'previous_convo': [],
    'subject': 'economics'}

chunks = []
with requests.get(url, headers=headers, data=json.dumps(payload), stream=True) as r:  
    for chunk in r.iter_content(1024):  
        print(chunk.decode('utf-8'))    
        chunks.append(chunk.decode('utf-8'))



print(''.join(chunks))