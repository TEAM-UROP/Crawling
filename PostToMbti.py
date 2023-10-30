import pandas as pd
import re

class PostToMbti:
    def __init__(self, csv_file_path, mbti_dic):
        self.df = pd.read_csv(csv_file_path) #본문 csv_file_path
        self.mbti_dic = mbti_dic #mbti 대응 사전
        self.valid_mbti = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                          'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']

    def toMbti(self, row):
        mbti_types = re.findall(r'[A-Z]+', row)
        if mbti_types:
            return mbti_types[0]
        converted_string = ''
        for char in row:
            if char in self.mbti_dic:
                converted_string += self.mbti_dic[char]
        return converted_string

    def process_data(self):
        # 'author' 열을 복사하여 'mbti' 열을 생성
        self.df['mbti'] = self.df['author'].apply(self.toMbti)

        # 필요한 MBTI 유형에 해당하는 행만 추출
        filtered_df = self.df[self.df['mbti'].isin(self.valid_mbti)]

        # 열 순서 변경 -> [본문] [댓글] [작성자] [mbti] 순으로
        new_order = ['contents', 'comments', 'author', 'mbti']
        filtered_df = filtered_df.reindex(columns=new_order)

        # 결과 출력
        print(filtered_df)
        print(f'유의미한 값의 비율: {(len(filtered_df) / len(self.df)) * 100:.2f}%')

if __name__ == "__main__":
    csv_file_path = "D:\\강의\\3-2\\학부연구참여(UROP)\\YongRullNewTown\\Crawling\\Post\\post_2023-10-29_14-15-34_(1 to 3).csv"
    mbti_dic = {
        '엥': 'EN', '뿌': 'F', '삐': 'P', '엔': 'EN', '인': 'IN',
        '티': 'T', '제': 'J', '잇': 'IS', '엣': 'ES', '팁': 'TP',
        '프': 'F', '푸': 'F', '1': 'I', '2': 'E', '피': 'P', '웬': 'EN',
        '줴': 'J', '쀼': 'F'
    }

    converter = PostToMbti(csv_file_path, mbti_dic)
    converter.process_data()
