import glob
import pandas as pd
import re
import json
from datetime import datetime


class PostToMbti:
    def __init__(self, csv_file_path, special_char):
        self.df = pd.read_csv(csv_file_path)  # 본문 csv_file_path
        with open(special_char, "r", encoding="UTF8") as json_file:
            self.special_char = json.load(json_file)
        self.valid_mbti = [
            "ISTJ",
            "ISFJ",
            "INFJ",
            "INTJ",
            "ISTP",
            "ISFP",
            "INFP",
            "INTP",
            "ESTP",
            "ESFP",
            "ENFP",
            "ENTP",
            "ESTJ",
            "ESFJ",
            "ENFJ",
            "ENTJ",
        ]

    def toMbti(self, row):
        mbti_types = re.findall(r"[A-Z]+", row)
        if mbti_types:
            return mbti_types[0]
        converted_string = ""
        for char in row:
            if char in self.special_char:
                converted_string += self.special_char[char]
        return converted_string

    def process_data(self):
        # 'author' 열을 복사하여 'mbti' 열을 생성
        self.df["mbti"] = self.df["author"].apply(self.toMbti)

        # 필요한 MBTI 유형에 해당하는 행만 추출
        filtered_df = self.df[self.df["mbti"].isin(self.valid_mbti)]

        # 열 순서 변경 -> [본문] [댓글] [mbti] 순으로
        new_order = ["mbti", "title", "contents"]
        filtered_df = filtered_df.reindex(columns=new_order)

        contents_copy = filtered_df.copy().drop("title", axis=1)

        # 결과 출력
        text_df = filtered_df.drop("contents", axis=1).rename(columns={"title": "text"})
        contents_copy = filtered_df.drop("title", axis=1).rename(
            columns={"contents": "text"}
        )
        post_df = pd.concat([text_df, contents_copy], axis=0)

        # csv 파일 저장
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"./data/labeled_post/labeled_post_{time}.csv"
        post_df.to_csv(file_name, index=False, encoding="utf-8-sig")

        # 결과 데이터프레임 출력
        print(post_df)
        print(f"유의미한 행의 개수: {len(contents_copy)} / {len(self.df)}")
        print(f"유의미한 값의 비율: {(len(post_df) / len(self.df)) * 0.5 * 100:.2f}%")


class CommentsToMbti:
    def __init__(self, csv_file_path, special_char):
        self.df = pd.read_csv(csv_file_path)  # 댓글 csv_file_path
        with open(special_char, "r", encoding="UTF8") as json_file:
            self.special_char = json.load(json_file)
        self.valid_mbti = [
            "ISTJ",
            "ISFJ",
            "INFJ",
            "INTJ",
            "ISTP",
            "ISFP",
            "INFP",
            "INTP",
            "ESTP",
            "ESFP",
            "ENFP",
            "ENTP",
            "ESTJ",
            "ESFJ",
            "ENFJ",
            "ENTJ",
        ]

    def toMbti(self, row):
        # 영어 MBTI로 변환 가능한 경우
        mbti_types = re.findall(r"[A-Z]+", row)
        if mbti_types:
            return mbti_types[0]
        # 한글 MBTI로 변환
        converted_string = ""
        for char in row:
            if char in self.special_char:
                converted_string += self.special_char[char]
        return converted_string

    def process_data(self):
        self.df["mbti"] = self.df["comments_writer"].apply(self.toMbti)
        filtered_df = self.df[self.df["mbti"].isin(self.valid_mbti)].drop(
            "comments_writer", axis=1
        )

        # 열 순서 변경
        new_order = ["mbti", "comments"]
        filtered_df = filtered_df.reindex(columns=new_order)

        # csv 파일 저장
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name2 = f"./data/labeled_comment/labeled_comment_{time}.csv"
        filtered_df.to_csv(file_name2, index=False, encoding="utf-8-sig")

        # 결과 출력
        print(filtered_df)
        print(f"유의미한 행의 개수: {len(filtered_df)} / {len(self.df)}")
        print(f"유의미한 값의 비율: {(len(filtered_df) / len(self.df)) * 100:.2f}%")


if __name__ == "__main__":
    csv_file_path_post_list = glob.glob("./data/post/*.csv")
    csv_file_path_comment_list = glob.glob("./data/comment/*.csv")
    special_char = "./data/special_char.json"

    for i in range(len(csv_file_path_post_list)):
        postfile = csv_file_path_post_list[i]
        converter_post = PostToMbti(postfile, special_char)
        converter_post.process_data()

    for j in range(len(csv_file_path_comment_list)):
        comments_file = csv_file_path_comment_list[j]
        converter_comment = CommentsToMbti(comments_file, special_char)
        converter_comment.process_data()
