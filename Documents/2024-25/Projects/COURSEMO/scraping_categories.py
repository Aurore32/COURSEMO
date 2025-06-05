from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup

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
                questions.append(stem + question[0])
                
            else:
                for second_order_question in question[1]:
                    questions.append(stem + question[0] + second_order_question)
    return questions

def getQuestions(html):
    soup = BeautifulSoup(html)
    questions = soup.find_all("div", class_="p-content")
    return questions

def getTexts(questions):
    texts = []
    for question in questions:
        subquestions = split(question)
        for subquestion in subquestions:
            texts = texts + [(subquestion, questions.index(question))]
    return texts

def getChapters(html):
    soup = BeautifulSoup(html)
    units = soup.find_all("div", class_="q-unit")
    difficulties = soup.find_all('div', class_='q-difficulty')
    units = [unit.get_text() for unit in units]
    difficulties = [difficulty.get_text() for difficulty in difficulties]
    return list(zip(units, difficulties))

subject_name = 'A level PHYSICS'

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
structured_button = driver.find_element(By.XPATH, "//li[text()='Structure question ']")
structured_button.click()
time.sleep(2)
filtered = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[9]/div/div/div/div')
filtered.click()
time.sleep(2)
search = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[11]/div/div/button[1]')
search.click()
search.click()

time.sleep(5)

combined_questions = []
combined_chapters =  []

for i in range(233):
    html = driver.page_source
    combined_questions = combined_questions + getQuestions(html)
    combined_chapters = combined_chapters + getChapters(html)
    pager = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[2]/div/div/span[2]/span[7]')
    pager.click()
    time.sleep(2)

assert len(combined_questions) == len(combined_chapters)

final_questions = []
for question in combined_questions:
    subquestions = split(question, False)
    for subquestion in subquestions:
        final_questions.append([subquestion, combined_chapters[combined_questions.index(question)][0], combined_chapters[combined_questions.index(question)][1]])

df = pd.DataFrame(final_questions, columns=['Question', 'Chapter', 'Difficulty'])
df.to_csv('al_physics_categories.csv')
