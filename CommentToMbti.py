import pandas as pd
import re


class CommentsToMbti:

    def __init__(self,csv_file_path, mbti_dic):
        self.df = pd.read_csv(csv_file_path)
        self.mbti_dic = mbti_dic
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
            if char in self.mbti_dic:
                converted_string += self.mbti_dic[char]
        return converted_string

    def process_data(self):
        self.df['mbti'] = self.df['comments_writer'].apply(self.toMbti)
        filtered_df = self.df[self.df['mbti'].isin(self.valid_mbti)]

        # 결과 출력
        print(filtered_df)
        print(f'유의미한 값의 비율: {(len(filtered_df) / len(self.df)) * 100:.2f}%')

if __name__ == "__main__":
    csv_file_path = "D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\comments\\comment_2023-10-29_14-15-34_(1 to 3).csv"
    mbti_dic = {
        '엥': 'EN', '뿌': 'F', '삐': 'P', '엔': 'EN', '인': 'IN',
        '티': 'T', '제': 'J', '잇': 'IS', '엣': 'ES', '팁': 'TP',
        '프': 'F', '푸': 'F', '1': 'I', '2': 'E', '피': 'P', '웬': 'EN',
        '줴': 'J', '쀼': 'F'
        }
    converter = CommentsToMbti(csv_file_path, mbti_dic)
    converter.process_data()