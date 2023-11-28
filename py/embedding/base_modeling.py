import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.optim as optim
from embedding import Embedding
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, n_classes=10):
        super(MyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.lstm = nn.LSTM(
            input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True
        )
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.n_classes)
        self.embedding = nn.Embedding(vocab_size, input_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        label = self.labels[idx]
        return data_sample, label

class TextModel:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trn_loader = None
        self.tst_loader = None
        self.val_loader = None

    def get_word_dict(self, corpus):
        df = pd.read_csv(corpus)
        # 텍스트를 토큰화하여 리스트로 변환
        corpus = [
            str(sentence).lower().split()
            for sentence in self.df["comments"]
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
    
    def data_loader(self):
        train_dataset = MyDataset(word_vectors[0])
        valid_dataset = MyDataset(word_vectors[1])
        test_dataset = MyDataset(word_vectors[2])

        
        self.trn_loader = DataLoader(train_dataset, batch_size=50, shuffle= True, drop_last=False)
        self.val_loader = DataLoader(valid_dataset, batch_size = 50, shuffle= False)
        self.tst_loader = DataLoader(test_dataset, batch_size=50, shuffle= False)
        
        
    def train(self, model, data_loader, optimizer, criterion, device):
        model.train()  # 모델을 학습모드로!
        trn_loss = 0
        for i, (label, text) in enumerate(data_loader):
            # Step 1. mini-batch에서 x,y 데이터를 얻고, 원하는 device에 위치시키기
            x = torch.LongTensor(text).to(device)
            y = torch.LongTensor(label).to(device)
            # Step 2. gradient 초기화
            optimizer.zero_grad()
            # Step 3. Forward Propagation
            y_pred_prob = model(x)
            # Step 4. Loss Calculation
            loss = criterion(y_pred_prob, y)
            # Step 5. Gradient Calculation (Backpropagation)
            loss.backward()
            # Step 6. Update Parameter (by Gradient Descent)
            optimizer.step()
            # Step 7. trn_loss 변수에 mini-batch loss를 누적해서 합산
            trn_loss += loss.item()
        # Step 8. 데이터 한 개당 평균 train loss
        avg_trn_loss = trn_loss / len(data_loader.dataset)
        return avg_trn_loss

    def evaluate(self, model, optimizer, criterion, device):
        model.eval()  # 모델을 평가모드로!
        eval_loss = 0
        results_pred = []
        results_real = []
        with torch.no_grad():  # evaluate()함수에는 단순 forward propagation만 할 뿐, gradient 계산 필요 X.
            for i, (label, text) in enumerate(self.tst_loader):
                # Step 1. mini-batch에서 x,y 데이터를 얻고, 원하는 device에 위치시키기
                x = torch.LongTensor(text).to(device)
                y = torch.LongTensor(label).to(device)
                # Step 2. Forward Propagation
                y_pred_prob = model(x)
                # Step 3. Loss Calculation
                loss = criterion(y_pred_prob, y)
                # Step 4. Predict label
                y_pred_label = torch.argmax(y_pred_prob, dim=1)
                # Step 5. Save real and predicted label
                results_pred.extend(y_pred_label.detach().cpu().numpy())
                results_real.extend(y.detach().cpu().numpy())
                # Step 6. eval_loss변수에 mini-batch loss를 누적해서 합산
                eval_loss += loss.item()
        # Step 7. 데이터 한 개당 평균 eval_loss와 accuracy구하기
        avg_eval_loss = eval_loss / len(self.tst_loader.dataset)
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
        word_vectors,
        csv,
        N_EPOCHS=10,
        LR=0.001,
        ):
        # 데이터셋 크기와 임베딩 차원 파악
        _, embedding_dim = word_vectors.shape
        # vocab = set(word for sentence in tokenized_corpus for word in sentence)
        VOCAB_SIZE = self.get_word_dict(csv)
        # 모델 인스턴스 생성
        model = MyLSTM(
            input_dim=embedding_dim, hidden_dim=50, vocab_size=VOCAB_SIZE, n_classes=10
        )
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss(reduction="sum")
        best_val_loss = float("inf")
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            trn_loss = self.train(
                model=model,
                data_loader=self.trn_loader,
                criterion=loss_func,
                optimizer=optimizer,
                device=self.device,
            )
            val_loss, accuracy = self.evaluate(
                model=model,
                data_loader=self.val_loader,
                criterion=loss_func,
                optimizer=optimizer,
                device=self.device,
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
