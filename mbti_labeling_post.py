import pandas as pd
import re
import json
import copy
import os

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', None)

class PostToMbti:
    def __init__(self, csv_file_path, special_char):
        self.df = pd.read_csv(csv_file_path) #본문 csv_file_path
        with open(special_char, 'r',encoding='UTF8') as json_file:
            self.special_char = json.load(json_file)
        self.valid_mbti = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                          'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']

    def toMbti(self, row):
        mbti_types = re.findall(r'[A-Z]+', row)
        if mbti_types:
            return mbti_types[0]
        converted_string = ''
        for char in row:
            if char in self.special_char:
                converted_string += self.special_char[char]
        return converted_string

    def process_data(self):
        # 'author' 열을 복사하여 'mbti' 열을 생성
        self.df['mbti'] = self.df['author'].apply(self.toMbti)

        # 필요한 MBTI 유형에 해당하는 행만 추출
        filtered_df = self.df[self.df['mbti'].isin(self.valid_mbti)]

        # 열 순서 변경 -> [본문] [댓글] [mbti] 순으로
        new_order = [ 'mbti', 'title','contents']
        filtered_df = filtered_df.reindex(columns=new_order)
        
        contents_copy = (filtered_df).copy().drop('title',axis=1)
        
        # 결과 출력

        text_df = filtered_df.drop('contents', axis=1).rename(columns={'title': 'text'})
        contents_copy = filtered_df.copy().drop('title', axis=1).rename(columns={'contents': 'text'})
        post_df = pd.concat([text_df, contents_copy], axis=0)

        #csv 파일 저장
        post_df.to_csv("D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\data\\newpost\\post_2023-10-30_15-20-40_(1 to 10).csv", index=False)

        # 결과 데이터프레임 c 출력
        print(post_df)
        print(f'유의미한 행의 개수: {len(contents_copy)} / {len(self.df)}' )
        print(f'유의미한 값의 비율: {(len(post_df) / len(self.df)) * 0.5 * 100:.2f}%')
        

        
if __name__ == "__main__":
    csv_file_path = "D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\data\\post\\post_2023-10-30_15-20-40_(1 to 10).csv"
    special_char = "D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\data\\special_char.json"

    converter = PostToMbti(csv_file_path, special_char)
    converter.process_data()
    
    