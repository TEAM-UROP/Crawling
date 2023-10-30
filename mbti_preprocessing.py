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

# 함수 정의
def clean_and_save_data(input_file_path):
    # 파일 불러오기
    post = pd.read_csv(input_file_path)
    comment = pd.read_csv(input_file_path)

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
    result = pd.DataFrame({'author': author, 'contents': new_text})
    result['contents'] = result['contents'].str.replace('\s+', ' ', regex=True)

    return result

# 처리할 파일 목록
input_files = ['./data/comment/comment_2023-10-30_15-20-40_(1 to 10).csv',
               './data/comment/comment_2023-10-30_15-26-54_(11 to 20).csv',
               './data/comment/comment_2023-10-30_15-38-32_(21 to 30).csv',
               './data/comment/comment_2023-10-30_15-46-01_(31 to 40).csv',
               './data/comment/comment_2023-10-30_15-53-01_(41 to 50).csv']

# 결과를 저장할 빈 데이터프레임 생성
final_result = pd.DataFrame(columns=['author', 'contents'])

# 각 파일을 정제하고 결과를 빈 데이터프레임에 추가
for input_file in input_files:
    cleaned_data = clean_and_save_data(input_file)
    final_result = pd.concat([final_result, cleaned_data])

# 모든 결과를 하나의 CSV 파일로 저장
final_result.to_csv('comment_data.csv', index=False)





    

