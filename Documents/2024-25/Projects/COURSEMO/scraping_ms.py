from bs4 import BeautifulSoup
import re
import datasets.hub
import pandas as pd
import datasets
import numpy as np

url = 'AL Econ MS only.html'

def strip(question):

    questions = []
    marks = []
    answers = []
    answers_new = []

    tags = question.find_all(['tr'])[1:]
    for tag in tags:
        if tag.find_all('strong') == []:
            pass
        else:
            texts = tag.find_all(['td'])
            if len(texts) < 3:
                pass
            else:
                answers.append(texts[1])
                marks.append(texts[2].get_text())

    for answer in answers:
        paragraphs = answer.find_all(['p','ul','li'])
        new_question = []
        for paragraph in paragraphs:
            if paragraph.find_all('strong') != []:
                new_question.append(paragraph.get_text())
            else:
                break
        answer_text = answer.get_text()
        for text in new_question:
            answer_text = answer_text.replace(text, '')
        answers_new.append(answer_text.lstrip())
        questions.append(' '.join(new_question))

    return list(zip(questions, marks, answers_new))

def getAnswers(url):
    with open(url, encoding='utf-8') as file:
        soup = BeautifulSoup(file)
        questions = soup.find_all("div", class_="p-answer")
        texts = []
        for question in questions:
            texts.append(question)
        return texts
        
answers = getAnswers(url)

answers_stripped = []
for i in range(0,161):
    answers_stripped = answers_stripped + strip(answers[i]) 
df = pd.DataFrame(np.array(answers_stripped), columns=['question', 'mark', 'answer'])
print(df['question'].tolist()[10:19])