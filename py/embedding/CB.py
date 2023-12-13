import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier


class CBModeling:
    def __init__(self, args, em_instance, word2vec_model, name):
        self.name = name
        self.em_instance = em_instance
        self.args = args
        self.corpus = []
        self.index = []
        for idx, item in enumerate(args.corpus):
            if str(item) != "['']" and item != ["nan"]:
                self.index.append(idx)
                self.corpus.append(item)
        self.word2vec_model = word2vec_model
        self.num_epochs = self.args.num_epochs

    def get_word_vector(self, word, model):
        try:
            return model.wv[word]
        except:
            return np.zeros(model.vector_size)

    def get_padding(self, corpus):
        end = 50
        word_vectors = []
        for sentence in corpus:
            temp = []
            for word in sentence[:end]:
                temp.append(self.get_word_vector(word, self.word2vec_model))
            padding_length = end - len(temp)
            if padding_length > 0:
                padding = [
                    np.zeros(self.word2vec_model.vector_size)
                    for _ in range(padding_length)
                ]
                temp.extend(padding)
            word_vectors.append(temp)
        return word_vectors

    def get_label(self):
        data = pd.read_csv(
            self.args.data,
            encoding="utf-8",
        )
        self.df = data[["mbti", "comments"]]
        self.df = self.df.iloc[self.index].reset_index(drop=True)
        for i in range(len(self.df)):
            if "E" in self.df.loc[i, "mbti"]:
                self.df.loc[i, "mbti"] = 0
            else:
                self.df.loc[i, "mbti"] = 1
        self.label = [int(i) for i in self.df["mbti"].values.tolist()]

    def get_dataloader(self):
        self.get_label()
        word_vectors = self.get_padding(self.corpus)
        input_tensor = word_vectors
        label_tensor = self.label

        d_len = len(self.df)
        self.trn_data = input_tensor[: int(d_len * 0.8)]
        self.trn_label = label_tensor[: int(d_len * 0.8)]
        self.val_data = input_tensor[int(d_len * 0.8) :]
        self.val_label = label_tensor[int(d_len * 0.8) :]

    def train_and_evaluate(self, epochs=5):
        self.get_dataloader()
        best_score = 0
        best_model = None
        x_train = np.array(self.trn_data)
        y_train = np.array(self.trn_label)
        x_val = np.array(self.val_data)
        y_val = np.array(self.val_label)
        catboost_model = CatBoostClassifier(iterations=1000, early_stopping_rounds=10)
        for epoch in range(epochs):
            x_train_re = []
            for j in x_train:
                x_train_re.append(np.mean(j, axis=1))
            x_val_re = []
            for j in x_val:
                x_val_re.append(np.mean(j, axis=1))
            catboost_model.fit(
                x_train_re, y_train, eval_set=(x_val_re, y_val), verbose=False
            )
            catboost_preds = catboost_model.predict(x_val_re)
            accuracy = accuracy_score(y_val, catboost_preds)
            if accuracy > best_score:
                best_score = accuracy
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Validation Accuracy: {best_score}")
            with open(f"CB_result({self.name}).txt", "a", encoding="utf-8") as f:
                f.write(
                    f"Epoch {epoch+1}/{epochs}\nValidation Accuracy: {best_score}\n"
                )
