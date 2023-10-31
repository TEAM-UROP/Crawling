import pandas as pd
import re
import json

class CommentsToMbti:

    def __init__(self, csv_file_path, special_char):
        self.df = pd.read_csv(csv_file_path) #댓글 csv_file_path
        with open(special_char, 'r',encoding='UTF8') as json_file:
            self.special_char = json.load(json_file)
        self.valid_mbti = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                          'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
    def toMbti(self,row):
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

        #열 순서 변경
        new_order = [ 'mbti','comments']
        filtered_df = filtered_df.reindex(columns=new_order)

        #csv 파일 저장
        filtered_df.to_csv("D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\data\\newcomment\\INTP ♧ ENTP_comment_2023-10-31_01-53-55_(1 to 30).csv", index=False)

        # 결과 출력
        print(filtered_df)
        print(f'유의미한 행의 개수: {len(filtered_df)} / {len(self.df)}' )
        print(f'유의미한 값의 비율: {(len(filtered_df) / len(self.df)) * 100:.2f}%')

if __name__ == "__main__":
    csv_file_path = "D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\data\\comment\\INTP ♧ ENTP_comment_2023-10-31_01-53-55_(1 to 30).csv"
    special_char = "D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\data\\special_char.json"

    converter = CommentsToMbti(csv_file_path, special_char)
    converter.process_data()