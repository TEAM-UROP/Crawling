import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
import LSTM as lstmpy


class LRModeling:
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
        self.df = None
        self.label = None
        self.train_loader = None
        self.val_loader = None
        self.trn_data = None
        self.trn_label = None
        self.val_data = None
        self.val_label = None
        self.model = LogisticRegression()

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

    def train(self, epochs=5):
        for epoch in range(epochs):
            for i in range(0, len(self.trn_data), self.args.batch_size):
                inputs = self.trn_data[i : i + self.args.batch_size]
                reduction = []
                for j in inputs:
                    reduction.append(np.mean(j, axis=1))
                labels = self.trn_label[i : i + self.args.batch_size]
                self.model.fit(reduction, labels)
            print(f"TRAIN : Epoch {epoch+1}/{epochs}")

    def evaluate(self):
        total_accuracy = 0
        cnt = 0
        for i in range(0, len(self.val_data), self.args.batch_size):
            inputs = self.val_data[i : i + self.args.batch_size]
            reduction = []
            for j in inputs:
                reduction.append(np.mean(j, axis=1))
            labels = self.val_label[i : i + self.args.batch_size]
            outputs = self.model.predict(reduction)
            with open("LR_pred.txt", "a", encoding="utf-8") as f:
                f.write(str(outputs) + "\n")
            accuracy = accuracy_score(labels, outputs)
            total_accuracy += accuracy
            cnt += 1
        avg_accuracy = total_accuracy / cnt
        return avg_accuracy

    def train_and_evaluate(self, epochs=5):
        for epoch in range(epochs):
            self.get_dataloader()
            print(f"Epoch {epoch+1}/{epochs}")
            with open(f"LR_result({self.name}).txt", "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch+1}/{epochs}\n")
            self.train(epochs=5)
            eval_accuracy = self.evaluate()
            print(f"Validation Accuracy: {eval_accuracy}")
            with open(f"LR_result({self.name}).txt", "a", encoding="utf-8") as f:
                f.write(f"Validation Accuracy: {eval_accuracy}\n")
