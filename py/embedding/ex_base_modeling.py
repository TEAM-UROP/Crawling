import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from embedding import Embedding
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MyLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        vocab_size,
        n_classes=2,
        num_layers=2,
        dropout=0.2,
    ):
        super(MyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.n_classes)
        # self.embedding = nn.Embedding(vocab_size, input_dim)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # embedded = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        # Pass through additional layers with dropout
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.dropout(out)
        out = F.relu(self.fc4(out))
        out = self.dropout(out)

        out = self.fc(out)
        return out


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # print(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        label = self.labels[idx]
        return data_sample, label


class TextModel:
    def __init__(self, args, word2vec_model):
        self.word2vec_model = word2vec_model
        self.args = args
        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trn_loader = None
        self.tst_loader = None
        self.val_loader = None
        self.train_word_vectors = None
        self.test_word_vectors = None
        self.val_word_vectors = None
        self.data = self.args.data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def get_word_dict(self):
        df = pd.read_csv(self.data)
        # 텍스트를 토큰화하여 리스트로 변환
        corpus = [
            str(sentence).lower().split()
            for sentence in df["comments"]
            if pd.notnull(sentence)
        ]
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)
        # 빈도가 높은 순서대로 상위 N개의 단어 선택 (N은 원하는 어휘 크기)
        higher = 100
        most_common_words = word_counts.most_common(higher)
        # 어휘 사전 구성 (단어를 인덱스로 매핑)
        word_to_index = {
            word: index for index, (word, _) in enumerate(most_common_words)
        }
        vocab_size = len(word_to_index)
        return vocab_size

    def prepare_data(self):
        X = self.args.corpus
        y = pd.read_csv(self.data)["mbti"].values
        labels = []
        for i in y:
            if "E" in str(i):
                labels.append(0)
            else:
                labels.append(1)
        # print(y)
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, labels, test_size=0.2, random_state=self.args.seed
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.args.seed
        )

    def get_word_vectors(self, sentences):
        # print(sentences)
        word_vectors = []
        for sentence in sentences:
            try:
                vectorized_sentence = []
                for word in sentence:
                    if word in self.word2vec_model.wv:
                        vectorized_sentence.append(self.word2vec_model.wv[word])
                    else:
                        vectorized_sentence.append(
                            np.zeros((self.word2vec_model.wv.vector_size))
                        )
                    # fix) np.zeros 말고 다른 방안을 생각해보자
            except:
                vectorized_sentence = np.zeros((self.word2vec_model.wv.vector_size))
            word_vectors.append(vectorized_sentence)
        return word_vectors

    def vectorize_data(self):
        self.prepare_data()
        self.train_word_vectors = self.get_word_vectors(self.X_train)
        self.val_word_vectors = self.get_word_vectors(self.X_val)
        self.test_word_vectors = self.get_word_vectors(self.X_test)

    def data_loader(self):
        train_dataset = MyDataset(self.train_word_vectors, self.y_train)
        valid_dataset = MyDataset(self.val_word_vectors, self.y_val)
        test_dataset = MyDataset(self.test_word_vectors, self.y_test)

        self.trn_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, drop_last=False
        )
        self.val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        self.tst_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def train(self, model, optimizer, criterion):
        model.train()
        trn_loss = 0
        for text, label in self.trn_loader:
            # x = torch.LongTensor(text).to(self.device)
            x = torch.tensor(text).to(self.device)
            # x = torch.LongTensor([i.numpy() for i in text]).to(self.device)
            y = torch.LongTensor(label).to(self.device)
            optimizer.zero_grad()
            y_pred_prob = model(x)
            loss = criterion(y_pred_prob, y)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item()
        avg_trn_loss = trn_loss / len(self.trn_loader.dataset)
        return avg_trn_loss

    def evaluate(self, model, criterion):
        model.eval()  # 모델을 평가모드로!
        eval_loss = 0
        results_pred = []
        results_real = []
        with torch.no_grad():
            for label, text in self.tst_loader:
                x = torch.LongTensor(text).to(self.device)
                y = torch.LongTensor(label).to(self.device)
                y_pred_prob = model(x)
                loss = criterion(y_pred_prob, y)
                y_pred_label = torch.argmax(y_pred_prob, dim=1)
                results_pred.extend(y_pred_label.detach().cpu().numpy())
                results_real.extend(y.detach().cpu().numpy())
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(self.val_loader.dataset)
        results_pred = np.array(results_pred)
        results_real = np.array(results_real)
        accuracy = np.sum(results_pred == results_real) / len(results_real)
        return avg_eval_loss, accuracy

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train_model(
        self,
        N_EPOCHS=10,
        LR=0.001,
    ):
        self.vectorize_data()
        self.data_loader()
        _, embedding_dim = self.word2vec_model.wv.vectors.shape
        VOCAB_SIZE = self.get_word_dict()
        # 모델 인스턴스 생성
        model = MyLSTM(
            input_dim=embedding_dim,
            hidden_dim=512,
            output_dim=1,
            vocab_size=VOCAB_SIZE,
            n_classes=2,
        )
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        best_val_loss = float("inf")
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            trn_loss = self.train(
                model=model,
                criterion=loss_func,
                optimizer=optimizer,
            )
            val_loss, accuracy = self.evaluate(
                model=model,
                criterion=loss_func,
            )
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\tTrain Loss: {trn_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {100 * accuracy:.3f}%"
            )


if __name__ == "__main__":
    csv = "./tokenized_0.csv"
    word_vectors = Embedding(csv).get_embedding_vector()
    text_model = TextModel()
    text_model.train_model(word_vectors, csv)
