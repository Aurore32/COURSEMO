import PyPDF2 as p2
import pandas as pd
import re
import string
import nltk
import numpy as np
import os

lemmatizer = nltk.stem.WordNetLemmatizer()

def scrape_questions(type: str, questions, pdf_path):
    pdf = open(pdf_path, 'rb')
    pdfread = p2.PdfReader(pdf)
    text = []
    for i in range(len(pdfread.pages)):
        pageinfo = pdfread.pages[i]
        rawinfo = pageinfo.extract_text()
        result = rawinfo.splitlines()
        text = text + result
    text = list(filter(('PhysicsAndMathsTutor.com').__ne__, text))

    question_start = []

    for i in range(len(text)):
        if text[i][:1].isdigit() and not text[i].isnumeric() and len(text[i]) > 8:
            question_start.append(i) 
    
    for i in range(len(question_start) - 1):
        start = question_start[i]
        end = question_start[i+1]
        question_text_list = text[start:end+1]
        question_text = ' '.join(question_text_list)
        question_text = clean_text(question_text)
        if len(question_text) < 2:
            pass
        else:
            question_text = ' '.join(question_text)
            labels = type.split(sep=',')
            question = [question_text, labels[0], labels[1]]
            questions.append(question)
        
    return questions

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'\n', '', text)
    translator = str.maketrans('' , '', string.punctuation)
    text = text.translate(translator)

    text_words = text.split()
    stopwords = nltk.corpus.stopwords.words('english')
    text_filtered= [word for word in text_words if word not in stopwords]
    text_filtered = [word for word in text_filtered if not (word.isalpha() and len(word) == 1)]

    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    valid_words = set(nltk.corpus.words.words())

    text_filtered = [word for word in text_filtered if word.lower() in valid_words]
    text_filtered = [lemmatizer.lemmatize(word) for word in text_filtered]
    text_filtered = list(set(text_filtered))

    return text_filtered

def remove_stopwords(data: pd.DataFrame):
    texts = data['text'].to_numpy().tolist()
    stopwords = nltk.corpus.stopwords.words('english')
    new_texts = []
    for text in texts:
        text_words = text.split()
        filtered_words = [word for word in text_words if word not in stopwords]
        new_texts.append(' '.join(filtered_words))
    data['text'] = new_texts
    data.to_csv('igcse-physics-new.csv')

def save_csv(type, questions, dir, name):
    files = os.listdir(dir)
    paths = [dir + '/' + file for file in files]
    for i in range(len(paths)):
        questions = scrape_questions(type=type[i], questions=questions, pdf_path=paths[i])  
    train_df = pd.DataFrame(np.array(questions), columns=['text','label1','label2'])
    train_df.to_csv(name + '.csv')
    return train_df

def make_combined_pdfs(dir, pages, output):
    files = os.listdir(dir)
    paths = [dir + '/' + file for file in files]
    merger = p2.PdfMerger()
    for path in paths:
        pdf = open(path, 'rb')
        merger.append(pdf, pages=(0,pages))
    merger.write(output)
