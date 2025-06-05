from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time

subject_name = 'A level CHEMISTRY'

driver = webdriver.Chrome()
driver.get('https://www.coursemo.com/account?client=portal&lang=zh')
username = driver.find_element(By.ID, ':r0:')
username.send_keys('19925170403')
password = driver.find_element(By.ID, 'password')
password.send_keys('199251')
login = driver.find_element(By.XPATH, '//*[@id="main"]/div/form/button')
login.click()
time.sleep(5)
driver.get('https://mo.coursemo.com/pct/exam/build-exam-ai')
time.sleep(5)
subject = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[1]/div/div/div/div')
subject.click()
time.sleep(2)
physics_button = driver.find_element(By.XPATH, "//li[text()='{} ']".format(subject_name))
physics_button.click()
time.sleep(2)
question_type = driver.find_element(By.XPATH, ' html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[3]/div/div/div/div')
question_type.click()
time.sleep(2)
structured_button = driver.find_element(By.XPATH, "//li[text()='Multiple choice ']")
structured_button.click()
time.sleep(2)
filtered = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[9]/div/div/div/div')
filtered.click()
time.sleep(2)
search = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[11]/div/div/button[1]')
search.click()
search.click()

time.sleep(5)
def getAnswers(url):
    soup = BeautifulSoup(url)
    questions = soup.find_all("div", class_="p-answer")
    texts = []
    for question in questions:
        texts.append(question)
    return texts
        
def getQuestions(url):
    soup = BeautifulSoup(url)   
    questions = soup.find_all("div", class_="p-content")
    texts = []
    for question in questions:
        texts.append(question)
    return texts

def split(soup: BeautifulSoup, explanation: bool):
    if soup.find_all('br') != None:
        for br in soup.find_all('br'):
            br.replace_with('\n')
    if soup.find_all('img') != None:
        for br in soup.find_all('img'):
            br.replace_with('\n(DIAGRAM)\n')
    base_question_tree = []
    first_order_questions = soup.find('ol')
    if first_order_questions != None:
        for question in first_order_questions.find_all('li'):
            first_order_question_tree = []
            second_order_questions = question.find('ol')
            second_order_question_tree = []
            if second_order_questions != None:
                for second_order_question in second_order_questions.find_all('li'):
                    second_order_question_tree.append(second_order_question.get_text())
                second_order_questions.decompose()
            first_order_question_tree.append(question.get_text())
            if second_order_question_tree == []:
                pass
            else:
                first_order_question_tree.append(second_order_question_tree)
            base_question_tree.append(first_order_question_tree)
        first_order_questions.decompose()
    stem = soup.get_text().strip()
    for j in ['[Total: {}]'.format(i) for i in range(1,21)]:
        stem = stem.replace(j, '')
    if explanation == True:
        stem = ''
    else:
        pass
    base_question_tree.append(stem)
    stem = base_question_tree.pop()
    base_question_tree = [stem] + base_question_tree
    base_question_tree = [tree for tree in base_question_tree if tree != ['']]
    questions = []
    numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    if len(base_question_tree) > 1:
        for question in base_question_tree[1:]:
            if len(question) == 1:
                questions.append((stem + question[0], chr(ord('a')+base_question_tree.index(question) - 1)))
                
            else:
                for second_order_question in question[1]:
                    questions.append((stem + question[0] + second_order_question, chr(ord('a')+base_question_tree.index(question) - 1) + '.{}'.format(numerals[question[1].index(second_order_question)])))
    return questions

def stripQuestions(questions: list):
    final_questions = []
    full_indices = []
    for i in range(len(questions)):
        question_list = split(questions[i], False)
        for k in range(len(question_list)):
            final_questions = final_questions + [[question_list[k][0].replace('\n', ' '), (i, question_list[k][1])]]
            full_indices.append((i, question_list[k][1]))

    return final_questions, full_indices

def rindex(lst, value):
    return len(lst) - 1 - list(reversed(lst)).index(value)

def stripMarkScheme(answers, full_indices):
    final_answer_list = []
    answer_indices = []
    indices = [full_indices[i][0] for i in range(len(full_indices))]
    print(indices)
    for i in range(len(answers)):
        if i in indices:
            answer = answers[i]
            if answer.find_all('br') != None:
                for br in answer.find_all('br'):
                    br.replace_with('\n')
            if answer.find_all('img') != None:
                for br in answer.find_all('br'):
                    br.replace_with('\n(DIAGRAM)\n')
            question_numbers = full_indices[indices.index(i):rindex(indices, i)+1]
            question_indices = [question_numbers[i][1] for i in range(len(question_numbers))]
            entries = answer.find_all('td')
            actual_answer_indices = [i for i in range(len(entries)) if entries[i].get_text().strip() in question_indices] + [len(entries)]
            if len(actual_answer_indices) == len(question_numbers) + 1:
                for j in range(len(actual_answer_indices) - 1):
                    mark_scheme = ' '.join([entry.get_text() for entry in entries[actual_answer_indices[j]+1:actual_answer_indices[j+1]]])
                    final_answer_list.append([mark_scheme.replace('\n', ' '), question_numbers[j]])
                    answer_indices.append(question_numbers[j])
                else:
                    pass
        else:
            pass

    return final_answer_list, answer_indices   

final_question_list = []

for i in range(100):
    html = driver.page_source
    questions = getQuestions(html) 
    answers = getAnswers(html)
    question_list, full_indices = stripQuestions(questions)
    answer_list, answer_indices = stripMarkScheme(answers, full_indices)
    print(answer_list)
    print(answer_indices)
   
    for question in question_list:
        if question[2] in answer_indices:
            question.insert(2, answer_list[answer_indices.index(question[2])][0])
            final_question_list.append(question)
    
    print(len(final_question_list))

    df = pd.DataFrame(final_question_list, columns=['question',  'answer', 'index'])
    df.to_csv('./chemistry/distillation/al_chemistry_structured_dataset.csv')


    pager = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[2]/div/div/span[2]/span[7]')
    pager.click()
    time.sleep(2)
