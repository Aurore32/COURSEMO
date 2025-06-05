from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup

def getQuestions(html):
    soup = BeautifulSoup(html)
    questions = soup.find_all("div", class_="p-content")
    questions = [question.get_text().strip() for question in questions]
    return questions

def getAnswers(html):
    soup = BeautifulSoup(html)
    answers = soup.find_all('div', class_='p-answer-option')
    answers = [answer.get_text().strip() for answer in answers]
    return answers

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

combined_questions = []
combined_answers =  []

for i in range(200):
    html = driver.page_source
    combined_questions = combined_questions + getQuestions(html)
    combined_answers = combined_answers + getAnswers(html)
    print(combined_answers)
    pager = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[2]/div/div/span[2]/span[7]')
    pager.click()
    time.sleep(2)

print(len(combined_questions))
print(len(combined_answers))

df = pd.DataFrame([])
df['Question'] = combined_questions
df['Mark Scheme'] = combined_answers
df.to_csv('al_physics_mcq_dataset.csv')
