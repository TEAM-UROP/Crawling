import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pyperclip
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv
import os
import datetime
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="MBTI Crawling")
    parser.add_argument("--naver_id", type=str, help="your naver id")
    parser.add_argument("--naver_password", type=str, help="your naver password")
    parser.add_argument("--start_page", type=str, help="start page number")
    parser.add_argument("--end_page", type=str, help="end page number")
    parser.add_argument("--post_dir", type=str, help="post directory")
    parser.add_argument("--comment_dir", type=str, help="comment directory")
    return parser


class Crwaling:
    def __init__(self):
        pass

    def naverLogin(self, naver_id, naver_password):
        driver.get("https://nid.naver.com/nidlogin.login")
        # enter id
        pyperclip.copy(naver_id)
        driver.find_element(By.ID, "id").send_keys(Keys.CONTROL, "v")
        # enter password
        pyperclip.copy(naver_password)
        driver.find_element(By.ID, "pw").send_keys(Keys.CONTROL, "v")
        # click the login button
        driver.find_element(By.ID, "log.login").click()

    def contentLoading(self):
        content = driver.find_elements(By.CSS_SELECTOR, ".se-fs-.se-ff-")
        inner_contents = []
        for i in content:
            inner_contents.append(i.text)
        content_list.append(inner_contents)
        return content_list

    def contentWriterLoading(self):
        try:
            content_writer = driver.find_element(
                By.XPATH,
                '//*[@id="app"]/div/div/div[2]/div[2]/div[2]/div/a/span[2]/strong',
            )
        except:
            content_writer = driver.find_element(
                By.XPATH,
                '//*[@id="app"]/div/div/div[2]/div[2]/div[3]/div/a/span[2]/strong',
            )
        content_writer_list.append(content_writer.text)
        return content_writer_list

    def commentsLoading(self):
        comments = driver.find_elements(By.XPATH, '//span[@class="text_comment"]')
        inner_contents = []
        if comments == "":
            inner_contents.append("")
        else:
            for i in comments:
                comment_list.append(i.text)
                inner_contents.append(i.text)
            comment_with_post_list.append(inner_contents)
        return comment_with_post_list, comment_list

    def commentsWriterLoading(self):
        comment_writer = driver.find_elements(By.CLASS_NAME, "comment_nickname")
        if comment_writer == "":
            comment_writer_list.append("")
        else:
            for i in comment_writer:
                comment_writer_list.append(i.text)
        return comment_writer_list

    def runCrawling(self, start_page, end_page):
        # move to the cafe
        driver.get("https://cafe.naver.com/mbticafe")
        # move to the first 사랑방
        driver.find_element(By.ID, "menuLink18").click()
        # move to the start page
        for page_num in range(start_page, end_page + 1):
            driver.switch_to.frame("cafe_main")
            driver.find_element(
                By.XPATH, f'//*[@id="main-area"]/div[6]/a[{page_num}]'
            ).click()
            time.sleep(1)
            for post_num in range(1, 16):
                if post_num != 1:
                    driver.switch_to.frame("cafe_main")
                # enter the post
                content_path = f'//*[@id="main-area"]/div[4]/table/tbody/tr[{post_num}]/td[1]/div[2]/div/a'
                driver.find_element(By.XPATH, content_path).click()
                time.sleep(1)
                # text crwaling
                content_list = self.contentLoading()
                content_writer_list = self.contentWriterLoading()
                comment_with_post_list, comment_list = self.commentsLoading()
                comment_writer_list = self.commentsWriterLoading()
                driver.back()
        # driver.quit()
        print("crawling is done")
        return (
            content_list,
            content_writer_list,
            comment_with_post_list,
            comment_list,
            comment_writer_list,
        )

    def getDataFrame(
        self,
        content_list,
        content_writer_list,
        comment_with_post_list,
        comment_list,
        comment_writer_list,
        post_dir,
        comment_dir,
        start_page,
        end_page,
    ):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # make dataframe of posts
        df_post = pd.DataFrame(
            {
                "contents": content_list,
                "author": content_writer_list,
                "comments": comment_with_post_list,
            }
        )
        df_post.to_csv(
            f"{post_dir}/post_{current_time}_({start_page} to {end_page}).csv",
            index=False,
            encoding="utf-8-sig",
        )
        # make dataframe of comments
        df_comment = pd.DataFrame(
            {"comments": comment_list, "comments_writer": comment_writer_list}
        )
        df_comment.to_csv(
            f"{comment_dir}/comment_{current_time}_({start_page} to {end_page}).csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("dataframe is made")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # set list
    content_list = []
    content_writer_list = []
    comment_with_post_list = []
    comment_list = []
    comment_writer_list = []

    # set args
    naver_id = args.naver_id
    naver_password = args.naver_password
    start_page = int(args.start_page)
    end_page = int(args.end_page)
    post_dir = args.post_dir
    comment_dir = args.comment_dir

    if not os.path.exists(post_dir):
        os.makedirs(post_dir)
    if not os.path.exists(comment_dir):
        os.makedirs(comment_dir)

    # set driver
    driver = webdriver.Chrome()

    # naver login
    crwaling = Crwaling()
    crwaling.naverLogin(naver_id, naver_password)

    # run crwaling
    (
        content_list,
        content_writer_list,
        comment_with_post_list,
        comment_list,
        comment_writer_list,
    ) = crwaling.runCrawling(start_page, end_page)

    # make dataframe
    crwaling.getDataFrame(
        content_list,
        content_writer_list,
        comment_with_post_list,
        comment_list,
        comment_writer_list,
        post_dir,
        comment_dir,
        start_page,
        end_page,
    )
