import pandas as pd
import re
import json
import os
from datetime import datetime

class PostToMbti:
    def __init__(self, csv_file_path, special_char):
        self.df = pd.read_csv(csv_file_path) # 본문 csv_file_path
        with open(special_char, 'r', encoding='UTF8') as json_file:
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
        new_order = ['mbti', 'title', 'contents']
        filtered_df = filtered_df.reindex(columns=new_order)

        contents_copy = filtered_df.copy().drop('title', axis=1)

        # 결과 출력
        text_df = filtered_df.drop('contents', axis=1).rename(columns={'title': 'text'})
        contents_copy = filtered_df.drop('title', axis=1).rename(columns={'contents': 'text'})
        post_df = pd.concat([text_df, contents_copy], axis=0)

        # csv 파일 저장
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'./data/newpost/post_file_{time}.csv'
        post_df.to_csv(file_name, index=False)

        # 결과 데이터프레임 출력
        print(post_df)
        print(f'유의미한 행의 개수: {len(contents_copy)} / {len(self.df)}')
        print(f'유의미한 값의 비율: {(len(post_df) / len(self.df)) * 0.5 * 100:.2f}%')

class CommentsToMbti:
    def __init__(self, csv_file_path, special_char):
        self.df = pd.read_csv(csv_file_path)  # 댓글 csv_file_path
        with open(special_char, 'r', encoding='UTF8') as json_file:
            self.special_char = json.load(json_file)
        self.valid_mbti = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                          'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']

    def toMbti(self, row):
        # 영어 MBTI로 변환 가능한 경우
        mbti_types = re.findall(r'[A-Z]+', row)
        if mbti_types:
            return mbti_types[0]
        # 한글 MBTI로 변환
        converted_string = ''
        for char in row:
            if char in self.special_char:
                converted_string += self.special_char[char]
        return converted_string

    def process_data(self):
        self.df['mbti'] = self.df['comments_writer'].apply(self.toMbti)
        filtered_df = self.df[self.df['mbti'].isin(self.valid_mbti)].drop("comments_writer", axis=1)

        # 열 순서 변경
        new_order = ['mbti', 'comments']
        filtered_df = filtered_df.reindex(columns=new_order)

        # csv 파일 저장
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name2 = f'./data/newcomment/comment_file_{time}.csv'
        filtered_df.to_csv(file_name2, index=False)

        # 결과 출력
        print(filtered_df)
        print(f'유의미한 행의 개수: {len(filtered_df)} / {len(self.df)}')
        print(f'유의미한 값의 비율: {(len(filtered_df) / len(self.df)) * 100:.2f}%')

if __name__ == "__main__":
    csv_file_path_post_list = ['./data/post/post_2023-10-30_15-20-40_(1 to 10).csv',
                               './data/post/post_2023-10-30_15-26-54_(11 to 20).csv',
                               './data/post/post_2023-10-30_15-38-32_(21 to 30).csv',
                               './data/post/post_2023-10-30_15-46-01_(31 to 40).csv',
                               './data/post/post_2023-10-30_15-53-01_(41 to 50).csv',
                               './data/post/INTP ♧ ENTP_post_2023-10-31_01-53-55_(1 to 30).csv']
    special_char_post = './data/special_char.json'
    csv_file_path_comment_list = ['./data/comment/comment_2023-10-30_15-20-40_(1 to 10).csv',
                                  './data/comment/comment_2023-10-30_15-26-54_(11 to 20).csv',
                                  './data/comment/comment_2023-10-30_15-38-32_(21 to 30).csv',
                                  './data/comment/comment_2023-10-30_15-46-01_(31 to 40).csv',
                                  './data/comment/comment_2023-10-30_15-53-01_(41 to 50).csv',
                                  './data/comment/INTP ♧ ENTP_comment_2023-10-31_01-53-55_(1 to 30).csv']
    special_char_comment = './data/special_char.json'

    for i in range(len(csv_file_path_post_list)):
        postfile = csv_file_path_post_list[i]
        converter_post = PostToMbti(postfile, special_char_post)
        converter_post.process_data()

    for j in range(len(csv_file_path_comment_list)):
        comments_file = csv_file_path_comment_list[j] 
        converter_comment = CommentsToMbti(comments_file, special_char_comment)
        converter_comment.process_data()