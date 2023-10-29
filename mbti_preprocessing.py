import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pyperclip
from selenium.webdriver.common.keys import Keys
#from dotenv import load_dotenv
import os
import datetime
import argparse
import re

# 파일 불러오기
post = pd.read_csv('./post/post_2023-10-29_14-13-32_(1 to 1).csv')
comment = pd.read_csv('./comment/comment_2023-10-29_14-13-32_(1 to 1).csv')

text = comment['comments']
author = comment['comments_writer']

new_text = []

# 기호 제거

def remove_punctuation(input_string):
    # 문자열에서 기호를 제거하는 정규 표현식 패턴
    pattern = r'[^\uAC00-\uD7A30-9a-zA-Z\s]'  
    # 정규 표현식 패턴을 사용하여 기호를 공백으로 대체
    try:
        result = re.sub(pattern, '', input_string)
    except:
        result = input_string
    return result

# 변환
for i in text:
    cleaned_string = remove_punctuation(i)
    new_text.append(cleaned_string)

# 줄바꿈 제거
result = pd.DataFrame({'author':author, 'contents':new_text})
result['contents'] = result['contents'].str.replace('\s+', ' ', regex=True)

# 결과임
print(result)