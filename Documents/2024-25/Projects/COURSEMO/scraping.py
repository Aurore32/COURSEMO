from bs4 import BeautifulSoup
import re
import datasets.hub
import pandas as pd
import datasets

url = 'AL Econ.html'

def strip(question):
    texts = []
    for question_tag in question.find_all(['li','p']):
        text = re.sub(pattern=r'([.])\1+', repl='.', string=question_tag.get_text())
        text.replace('\xa0', '\n')
        texts.append(text)
    return ' '.join(texts)

def getQuestions(url):
    with open(url, encoding='utf-8') as file:
        soup = BeautifulSoup(file)
        questions = soup.find_all("div", class_="p-content")
        texts = []
        for question in questions:
            texts.append(strip(question))
        return texts

def getAnswers(url):
    with open(url, encoding='utf-8') as file:
        soup = BeautifulSoup(file)
        questions = soup.find_all("div", class_="p-explain")
        texts = []
        for question in questions:
            texts.append(strip(question))
        return texts
        
questions = getQuestions(url)
answers = getAnswers(url)

df = pd.DataFrame([])
df['Instruction'] = ['You are an A-Level economics teacher with full knowledge of the A-Level economics syllabus. Your task is to write a sample essay that will obtain top marks according to the A-Level mark scheme on the given essay question.\n' for i in range(len(questions))]
df['Question'] = questions
df['Answer'] = answers

print(df)

df.to_csv('al_econ_train.csv')