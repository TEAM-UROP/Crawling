import pandas as pd
import re
from datetime import datetime
from glob import glob


def remove_punctuation(input_string):
    # 문자열에서 기호를 제거하는 정규 표현식 패턴
    pattern = r"[^\uAC00-\uD7A30-9a-zA-Z\s]"
    # 정규 표현식 패턴을 사용하여 기호를 공백으로 대체
    try:
        result = re.sub(pattern, "", input_string)
    except:
        result = input_string
    return result


def clean_and_save_data(input_file_path):
    comment = pd.read_csv(input_file_path)
    mbti = comment["mbti"]
    comments = comment["comments"]
    new_text = []
    for i in comments:
        cleaned_string = remove_punctuation(i)
        new_text.append(cleaned_string)
    # 줄바꿈 제거
    result = pd.DataFrame({"mbti": mbti, "comments": new_text})
    result["comments"] = result["comments"].str.replace("\s+", " ", regex=True)
    return result


def clean_and_save_data2(input_file_path):
    post = pd.read_csv(input_file_path)
    mbti = post["mbti"]
    text = post["text"]
    new_text = []
    for i in text:
        cleaned_string = remove_punctuation(i)
        new_text.append(cleaned_string)
    result = pd.DataFrame({"mbti": mbti, "text": new_text})
    result["text"] = result["text"].str.replace("\s+", " ", regex=True)
    return result


if __name__ == "__main__":
    input_files_comment = glob("./data/labeled_comment/*.csv")
    input_files_post = glob("./data/labeled_post/*.csv")

    # 결과를 저장할 빈 데이터프레임 생성
    final_result_comment = pd.DataFrame(columns=["mbti", "comments"])
    final_result_post = pd.DataFrame(columns=["mbti", "text"])

    # 각 파일을 정제하고 결과를 빈 데이터프레임에 추가
    for input_file in input_files_comment:
        cleaned_data_comment = clean_and_save_data(input_file)
        final_result_comment = pd.concat([final_result_comment, cleaned_data_comment])

    for input_file in input_files_post:
        cleaned_data_post = clean_and_save_data2(input_file)
        final_result_post = pd.concat([final_result_post, cleaned_data_post])

    # 모든 결과를 하나의 CSV 파일로 저장
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    comments_name = f"./data/pre_comment/pre_comment_{time}.csv"
    post_name = f"./data/pre_post/pre_post_{time}.csv"
    
    final_result_comment.to_csv(comments_name, index=False, encoding="utf-8")
    final_result_post.to_csv(post_name, index=False, encoding="utf-8")
