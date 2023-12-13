import pandas as pd
from konlpy.tag import Kkma, Okt, Hannanum


class Tokenizer:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args.data, encoding="utf-8")
        self.okt = Okt()
        self.kkma = Kkma()
        self.hannanum = Hannanum()
        self.data["comments"] = self.data["comments"].astype(str)

    def tokenizing(self):
        okt_morphs = self.data["comments"].apply(
            lambda row: " ".join(self.okt.morphs(row))
        )
        okt_nouns = self.data["comments"].apply(
            lambda row: " ".join(self.okt.nouns(row))
        )
        kkma_morphs = self.data["comments"].apply(
            lambda row: " ".join(self.kkma.morphs(row))
        )
        kkma_nouns = self.data["comments"].apply(
            lambda row: " ".join(self.kkma.nouns(row))
        )
        hannaum_morphs = self.data["comments"].apply(
            lambda row: " ".join(self.hannanum.morphs(row))
        )
        hannaum_nouns = self.data["comments"].apply(
            lambda row: " ".join(self.hannanum.morphs(row))
        )
        res = (
            (okt_morphs, "okt_morphs"),
            (okt_nouns, "okt_nouns"),
            (kkma_morphs, "kkma_morphs"),
            (kkma_nouns, "kkma_nouns"),
            (hannaum_morphs, "hannaum_morphs"),
            (hannaum_nouns, "hannaum_nouns"),
        )
        return res
