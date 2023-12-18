import os
import pandas as pd
from konlpy.tag import Kkma, Okt, Hannanum


class Tokenizer:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args.data, encoding="utf-8")
        self.okt = Okt()
        self.kkma = Kkma()
        self.hannanum = Hannanum()
        self.filename = os.path.basename(self.args.data).replace(".csv", "")
        self.data["text"] = self.data["text"].astype(str)

    def tokenizing(self):
        os.makedirs("token", exist_ok=True)
        try:
            if not os.path.isfile(f"token/okt_morphs_{self.filename}.csv"):
                okt_morphs = self.data["text"].apply(
                    lambda row: " ".join(self.okt.morphs(row))
                )
                okt_morphs.to_csv(
                    f"token/okt_morphs_{self.filename}.csv",
                    encoding="utf-8",
                    index=False,
                )
            elif os.path.isfile(f"token/okt_morphs_{self.filename}.csv"):
                okt_morphs = pd.read_csv(
                    f"token/okt_morphs_{self.filename}.csv", encoding="utf-8"
                )
        except:
            print("okt_morphs error")
        try:
            if not os.path.isfile(f"token/okt_nouns_{self.filename}.csv"):
                okt_nouns = self.data["text"].apply(
                    lambda row: " ".join(self.okt.nouns(row))
                )
                okt_nouns.to_csv(
                    f"token/okt_nouns_{self.filename}.csv",
                    encoding="utf-8",
                    index=False,
                )
            elif os.path.isfile(f"token/okt_nouns_{self.filename}.csv"):
                okt_nouns = pd.read_csv(
                    f"token/okt_nouns_{self.filename}.csv", encoding="utf-8"
                )
        except:
            print("okt_nouns error")
        try:
            if not os.path.isfile(f"token/kkma_morphs_{self.filename}.csv"):
                kkma_morphs = self.data["text"].apply(
                    lambda row: " ".join(self.kkma.morphs(row))
                )
                kkma_morphs.to_csv(
                    f"token/kkma_morphs_{self.filename}.csv",
                    encoding="utf-8",
                    index=False,
                )
            elif os.path.isfile(f"token/kkma_morphs_{self.filename}.csv"):
                kkma_morphs = pd.read_csv(
                    f"token/kkma_morphs_{self.filename}.csv", encoding="utf-8"
                )
        except:
            print("kkma_morphs error")
        try:
            if not os.path.isfile(f"token/kkma_nouns_{self.filename}.csv"):
                kkma_nouns = self.data["text"].apply(
                    lambda row: " ".join(self.kkma.nouns(row))
                )
                kkma_nouns.to_csv(
                    f"token/kkma_nouns_{self.filename}.csv",
                    encoding="utf-8",
                    index=False,
                )
            elif os.path.isfile(f"token/kkma_nouns_{self.filename}.csv"):
                kkma_nouns = pd.read_csv(
                    f"token/kkma_nouns_{self.filename}.csv", encoding="utf-8"
                )
        except:
            print("kkma_nouns error")
        try:
            if not os.path.isfile(f"token/hannanum_morphs_{self.filename}.csv"):
                hannanum_morphs = self.data["text"].apply(
                    lambda row: " ".join(self.hannanum.morphs(row))
                )
                hannanum_morphs.to_csv(
                    f"token/hannanum_morphs_{self.filename}.csv",
                    encoding="utf-8",
                    index=False,
                )
            elif os.path.isfile(f"token/hannanum_morphs_{self.filename}.csv"):
                hannanum_morphs = pd.read_csv(
                    f"token/hannanum_morphs_{self.filename}.csv", encoding="utf-8"
                )
        except:
            print("hannanum_morphs error")
        try:
            if not os.path.isfile(f"token/hannanum_nouns_{self.filename}.csv"):
                hannanum_nouns = self.data["text"].apply(
                    lambda row: " ".join(self.hannanum.nouns(row))
                )
                hannanum_nouns.to_csv(
                    f"token/hannanum_nouns_{self.filename}.csv",
                    encoding="utf-8",
                    index=False,
                )
            elif os.path.isfile(f"token/hannanum_nouns_{self.filename}.csv"):
                hannanum_nouns = pd.read_csv(
                    f"token/hannanum_nouns_{self.filename}.csv", encoding="utf-8"
                )
        except:
            print(f"token/hannanum_nouns_{self.filename}.csv")
        # okt_nouns = self.data["text"].apply(
        #     lambda row: " ".join(self.okt.nouns(row))
        # )
        # kkma_morphs = self.data["text"].apply(
        #     lambda row: " ".join(self.kkma.morphs(row))
        # )
        # kkma_nouns = self.data["text"].apply(
        #     lambda row: " ".join(self.kkma.nouns(row))
        # )
        # hannanum_morphs = self.data["text"].apply(
        #     lambda row: " ".join(self.hannanum.morphs(row))
        # )
        # hannanum_nouns = self.data["text"].apply(
        #     lambda row: " ".join(self.hannanum.morphs(row))
        # )
        res = (
            (okt_morphs, "okt_morphs"),
            (okt_nouns, "okt_nouns"),
            (kkma_morphs, "kkma_morphs"),
            (kkma_nouns, "kkma_nouns"),
            (hannanum_morphs, "hannanum_morphs"),
            (hannanum_nouns, "hannanum_nouns"),
        )
        return res
