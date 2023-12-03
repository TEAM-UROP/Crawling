import pandas as pd
from konlpy.tag import Kkma, Okt, Hannanum
from embedding import Embedding


class Tokenizer:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args.data, encoding="utf-8")
        self.okt = Okt()
        self.kkma = Kkma()
        self.hannanum = Hannanum()

    def tokenizing(self):
        self.data["comments"] = self.data["comments"].astype(str)
        mbti = self.data[['mbti']]
        # tok_okt_morphs = self.data["comments"].apply(
        #     lambda row: " ".join(self.okt.morphs(row))
        #  )

        # tok_okt_pos = self.data["comments"].apply(
        #     lambda row: " ".join(
        #         ["{}/{}".format(word, pos) for word, pos in self.okt.pos(row)]
        #     )
        # )

        tok_okt_nouns = self.data["comments"].apply(
            lambda row: " ".join(self.okt.nouns(row))
        )

        # tok_kkma_morphs = self.data["comments"].apply(
        #     lambda row: " ".join(self.kkma.morphs(row))
        # )

        # tok_kkma_pos = self.data["comments"].apply(
        #     lambda row: " ".join(
        #         ["{}/{}".format(word, pos) for word, pos in self.kkma.pos(row)]
        #     )
        # )

        # tok_kkma_nouns = self.data["comments"].apply(
        #     lambda row: " ".join(self.kkma.nouns(row))
        # )

        # tok_hannaum_morphs = self.data["comments"].apply(
        #     lambda row: " ".join(self.hannanum.morphs(row))
        # )

        # tok_hannaum_pos = self.data["comments"].apply(
        #     lambda row: " ".join(
        #         ["{}/{}".format(word, pos) for word, pos in self.hannanum.pos(row)]
        #     )
        # )

        # tok_hannaum_nouns = self.data["comments"].apply(
        #     lambda row: " ".join(self.hannanum.morphs(row))
        # )

        res = (
            # tok_okt_morphs,
            # tok_okt_pos,
            tok_okt_nouns,
            # tok_kkma_morphs,
            # tok_kkma_pos,
            # tok_kkma_nouns,
            # tok_hannaum_morphs,
            # tok_hannaum_pos,
            # tok_hannaum_nouns,
        )
        aba = []
        for i in range(len(res)):
            new_res = pd.DataFrame(res[i])
            # new_res=new_res.transpose()
            real_res = pd.concat([new_res,mbti],axis=1)
            aba.append(real_res)
            return aba


if __name__ == "__main__":
    # pre = Tokenizer()
    # res = pre.tokenizing()
    # for idx, i in enumerate(res):
    #     i.to_csv(f"tokenized_{idx}.csv", index=False, encoding="utf-8-sig")
    #     break
    embedding = Embedding("./tokenized_0.csv")
