import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size),
        )

    def forward(self, input):
        batch_size = input.size(0)
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        output = self.fc(lstm_out[:, -1, :])
        output = torch.sigmoid(output)
        with open("out.txt", "a", encoding="utf-8") as f:
            f.write(str(output) + "\n")
        return output


class LSTMModeling:
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
        self.input_size = self.word2vec_model.vector_size
        self.hidden_size = 16
        self.output_size = 1
        self.batch_size = self.args.batch_size
        self.lstm_model = LSTMModel(
            self.input_size, self.hidden_size, self.output_size, self.batch_size
        )
        self.num_epochs = self.args.num_epochs
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.args.lr)
        self.df = None
        self.label = None
        self.train_loader = None
        self.val_loader = None

    def get_word_vector(self, word, model):
        try:
            return model.wv[word]
        except:
            return np.zeros(self.input_size)

    def get_padding(self, corpus):
        end = 50
        word_vectors = []
        for sentence in corpus:
            temp = []
            for word in sentence[:end]:
                temp.append(self.get_word_vector(word, self.word2vec_model))
            padding_length = end - len(temp)
            if padding_length > 0:
                padding = [np.zeros(self.input_size) for _ in range(padding_length)]
                temp.extend(padding)
            word_vectors.append(temp)
        return word_vectors

    def get_label(self):
        data = pd.read_csv(
            self.args.data,
            encoding="utf-8",
        )
        self.df = data[["mbti", "text"]]
        self.df = self.df.iloc[self.index].reset_index(drop=True)
        for i in range(len(self.df)):
            if "E" in self.df.loc[i, "mbti"]:
                self.df.loc[i, "mbti"] = 1
            else:
                self.df.loc[i, "mbti"] = 0
        self.label = [int(i) for i in self.df["mbti"].values.tolist()]

    def get_dataloader(self):
        self.get_label()
        word_vectors = self.get_padding(self.corpus)
        input_tensor = torch.tensor(word_vectors, dtype=torch.float32)
        label_tensor = torch.LongTensor(self.label)

        d_len = len(self.df)
        trn_data = input_tensor[: int(d_len * 0.8)]
        trn_label = label_tensor[: int(d_len * 0.8)]
        val_data = input_tensor[int(d_len * 0.8) :]
        val_label = label_tensor[int(d_len * 0.8) :]

        trn_dataset = MyDataset(trn_data, trn_label)
        val_dataset = MyDataset(val_data, val_label)

        self.train_loader = DataLoader(
            trn_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=True
        )

    def train(self, epochs=5):
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.lstm_model(inputs)
                loss = self.criterion(outputs.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(
                f"TRAIN : Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(self.train_loader)}"
            )
            with open(f"LSTM_result({self.name}).txt", "a", encoding="utf-8") as f:
                f.write(
                    f"TRAIN : Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(self.train_loader)}\n"
                )

    def evaluate(self):
        self.lstm_model.eval()
        total_loss = 0.0
        # total_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.lstm_model(inputs)
                loss = self.criterion(outputs.squeeze(), labels.float())
                total_loss += loss.item()
                preds = outputs.round()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
                with open("pred.txt", "a", encoding="utf-8") as f:
                    f.write(str(preds) + "\n")
                # ans = preds == labels.data
                # for i in range(len(ans[0])):
                    # if ans[0][i] == True:
                        # total_corrects += 1
        avg_loss = total_loss / len(self.val_loader)
        # accuracy = total_corrects / len(self.val_loader.dataset)
        f1 = f1_score(all_labels, all_preds)
        # print(f"Loss: {avg_loss}, Accuracy: {accuracy}")
        return avg_loss, f1

    def train_and_evaluate(self, epochs=5):
        for epoch in range(epochs):
            self.get_dataloader()
            print(f"Epoch {epoch+1}/{epochs}")
            with open(f"LSTM_result({self.name}).txt", "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch+1}/{epochs}\n")
            self.train(epochs=5)
            # eval_loss, eval_accuracy = self.evaluate()
            eval_loss, eval_f1 = self.evaluate()
            print(f"Validation Loss: {eval_loss}, Validation F1 Score: {eval_f1}")
            with open(f"LSTM_result({self.name}).txt", "a", encoding="utf-8") as f:
                f.write(
                    f"Validation Loss: {eval_loss}, Validation F1 Score: {eval_f1}\n"
                )
