import pandas as pd
from embedding import Embedding
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
        return output


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_word_vector(word, model):
    try:
        return model.wv[word]
    except:
        return np.zeros(input_size)


def train(model, criterion, optimizer, train_loader, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # train_loader에는 입력 데이터와 레이블이 있는 데이터가 포함되어야 합니다.
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            # BCELoss를 사용하므로 레이블을 float로 변환합니다.

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def evaluate(model, criterion, eval_loader):
    model.eval()
    total_loss = 0.0
    total_corrects = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
            preds = outputs.round()
            total_corrects += torch.sum(preds == labels.data)
    avg_loss = total_loss / len(eval_loader)
    accuracy = total_corrects.double() / len(eval_loader.dataset)
    print(f"Loss: {avg_loss}, Accuracy: {accuracy}")
    return avg_loss, accuracy


def train_and_evaluate(
    model, criterion, optimizer, train_loader, eval_loader, epochs=5
):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train(model, criterion, optimizer, train_loader, epochs=1)
        eval_loss, eval_accuracy = evaluate(model, criterion, eval_loader)
        print(f"Validation Loss: {eval_loss}, Validation Accuracy: {eval_accuracy}")


if __name__ == "__main__":
    data = pd.read_csv(
        "C:/Users/user/OneDrive/문서/dev/UROP/Crawling/data/pre_comment/pre_comment_2023-11-12_21-04-00.csv",
        encoding="utf-8",
    )
    token = pd.read_csv(
        "C:/Users/user/OneDrive/문서/dev/UROP/Crawling/real_comment (1).csv",
        encoding="utf-8",
    )
    df = data[["mbti", "comments"]]
    df = df.dropna().reset_index(drop=True)
    token = token.dropna().reset_index(drop=True)

    for i in range(len(df)):
        if "E" in df.loc[i, "mbti"]:
            df.loc[i, "mbti"] = 0
        else:
            df.loc[i, "mbti"] = 1

    em = Embedding(sereis=token["comments"])
    word2vec_model = em.get_embedding_model()

    input_size = word2vec_model.vector_size
    hidden_size = 128
    output_size = 1
    batch_size = 32
    lstm_model = LSTMModel(input_size, hidden_size, output_size, batch_size)

    end = 100
    word_vectors = []
    for sentence in em.corpus:
        temp = []
        for word in sentence[:end]:
            temp.append(get_word_vector(word, word2vec_model))
        padding_length = end - len(temp)
        if padding_length > 0:
            padding = [
                np.zeros(word2vec_model.vector_size) for _ in range(padding_length)
            ]
            temp.extend(padding)
        word_vectors.append(temp)

    input_tensor = torch.tensor(word_vectors, dtype=torch.float32)
    label = [int(i) for i in df["mbti"].values.tolist()]
    label_tensor = torch.LongTensor(label)
    # print(len(df) * 0.8)
    trn_data = input_tensor[: int(len(df) * 0.8)]
    trn_label = label_tensor[: int(len(df) * 0.8)]
    val_data = input_tensor[int(len(df) * 0.8) :]
    val_label = label_tensor[int(len(df) * 0.8) :]

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    trn_dataset = MyDataset(trn_data, trn_label)
    val_dataset = MyDataset(val_data, val_label)

    train_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train_and_evaluate(
        lstm_model, criterion, optimizer, train_loader, val_loader, epochs=10
    )
