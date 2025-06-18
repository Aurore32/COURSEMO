from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup

# 用Selenium控制Google Chrome访问COURSEMO数据库 找到你想要的科目+题型 然后实现翻页
# 在每一页题目上用BeautifulSoup提取题目信息
# 以下代码提取选择题信息 结构题稍微复杂一些

def getQuestions(html):
    soup = BeautifulSoup(html)
    questions = soup.find_all("div", class_="p-content") # 题目本体
    questions = [question.get_text().strip() for question in questions]
    return questions

def getAnswers(html):
    soup = BeautifulSoup(html)
    answers = soup.find_all('div', class_='p-answer-option') # 正确答案
    answers = [answer.get_text().strip() for answer in answers]
    return answers

subject_name = 'IGCSE PHYSICS' # 换成你的科目

driver = webdriver.Chrome()
driver.get('https://www.coursemo.com/account?client=portal&lang=zh')
username = driver.find_element(By.ID, ':r0:')
username.send_keys('19925170403') # 用这个账号就好啦 反正人家也不会知道的
password = driver.find_element(By.ID, 'password')
password.send_keys('199251')
login = driver.find_element(By.XPATH, '//*[@id="main"]/div/form/button')
login.click() # 登录
time.sleep(5)
driver.get('https://mo.coursemo.com/pct/exam/build-exam-ai') # 到数据库里
time.sleep(5)
subject = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[1]/div/div/div/div')
subject.click() # 点击某个按钮 到底是哪个我也忘了
time.sleep(2)
physics_button = driver.find_element(By.XPATH, "//li[text()='{} ']".format(subject_name))
physics_button.click() # 点击科目名称按钮
time.sleep(2)
question_type = driver.find_element(By.XPATH, ' html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[3]/div/div/div/div')
question_type.click() # 点击题型按钮 改成你的题型（选择题/结构题）
time.sleep(2)
structured_button = driver.find_element(By.XPATH, "//li[text()='Multiple choice ']")
structured_button.click() # 点击题型按钮 改成你的题型（选择题/结构题）
time.sleep(2)
filtered = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[9]/div/div/div/div')
filtered.click() # 过滤按钮
time.sleep(2)
search = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[1]/div[2]/div/div[11]/div/div/button[1]')
search.click()
search.click() # 搜索按钮

time.sleep(5)

combined_questions = []
combined_answers =  []

for i in range(100): # 1-100页的题目
    # 用BeautifulSoup提取题目信息
    html = driver.page_source
    combined_questions = combined_questions + getQuestions(html)
    combined_answers = combined_answers + getAnswers(html)
    pager = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div[2]/div[2]/div[2]/div/div/div[1]/div[3]/div[2]/div/div/span[2]/span[7]')
    print(len(combined_questions))
    print(len(combined_answers))
    df = pd.DataFrame([])
    df['Question'] = combined_questions
    df['Mark Scheme'] = combined_answers
    df.to_csv('./physics/distillation/igcse_physics_mcq_dataset.csv')

    pager.click()
    time.sleep(2)



