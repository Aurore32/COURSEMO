import requests
from io import BytesIO
from urllib.request import urlretrieve
import os
from urllib.parse import unquote, urlparse
from pathlib import Path
import pandas as pd
import fitz
import pdfplumber
from pylatexenc import latex2text
import re
from tqdm import tqdm

def clean_paper_text(text):
    # Remove header/footer patterns
    patterns = [
        r"MARK SCHEME FOR THE .*? PAPER",
        r"Cambridge International Advanced Subsidiary and Advanced Level",
        r"MAXIMUM RAW MARK \d+",
        r"Page \d+ of \d+",
        r"9702/\d+ .*? (May/June|March|October/November) 20\d{2}",
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove question metadata lines
    text = re.sub(r"^Question \d+.*$\n?", "", text, flags=re.MULTILINE)
    
    return text.strip()


def process_pdf_url(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def process_pdf_url(path):
    text = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

ms_urls = []
for year in range(10,25):
    ms_urls += [
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_w{}_ms_53.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_w{}_ms_51.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_s{}_ms_53.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_s{}_ms_51.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_m{}_ms_52.pdf'.format(year)
    ]
for year in range(7, 10):
    ms_urls += [
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_s0{}_ms_5.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_w0{}_ms_5.pdf'.format(year)
    ]

qp_urls = []
for year in range(10,25):
    ms_urls += [
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_w{}_qp_53.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_w{}_qp_51.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_s{}_qp_53.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_s{}_qp_51.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_m{}_qp_52.pdf'.format(year)
    ]
for year in range(7, 10):
    ms_urls += [
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_s0{}_qp_5.pdf'.format(year),
        'https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/9702_w0{}_qp_5.pdf'.format(year)
    ]

def replace_cid_codes(text):
    cid_map = {
    "(cid:129)": "•",  # Bullet points
    "(cid:183)": "·",  # Middle dot
    "(cid:176)": "°",  # Degree symbol
    # Add mappings from your PDF's common CIDs
    }
    text = re.sub(r"\(cid:\d+\)", lambda m: cid_map.get(m.group(), ""), text)


def full_clean_pipeline(text):
    # Step 2: Clean headers/footers
    text = clean_paper_text(text)
    
    # Step 3: Fix CIDs
    text = replace_cid_codes(text)
            
    return text


dirs = os.listdir('./physics/pdfs_ms')
questions = []
markschemes = []
for url in tqdm(dirs):
    path = './physics/pdfs_ms/{}'.format(url.replace('https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/', ''))
    try:
        if 'qp' in path:
            ms_path = './physics/pdfs_ms/{}'.format(url.replace('https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/', '').replace('qp','ms'))
            questions.append(process_pdf_url(path).split('..............................')[0])
            if int(path.split('9702')[1].split('_')[1].replace('s','').replace('w','').replace('m','')) > 16:
                markschemes.append(process_pdf_url(ms_path).split('2(a)')[0])
            else:
                markschemes.append(process_pdf_url(ms_path).split('[Total: 15 marks]')[0])
        else:
            pass
    except:
        pass


df =  pd.DataFrame([])
df['Question'] = questions
df['Mark Scheme'] = markschemes
df.to_csv('./physics/distillation/al_physics_experiment_dataset.csv', encoding='utf-8')