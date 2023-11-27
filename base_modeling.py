import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import optuna
from gensim.models import Word2Vec
import torch.optim as optim
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, n_classes=10):
        super(MyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.n_classes)
        self.embedding = nn.Embedding(vocab_size, input_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class TextModel:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self, model, data_loader, optimizer, criterion, device):
        model.train() # 모델을 학습모드로!
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

    def evaluate(self, model, data_loader, optimizer, criterion, device):
        model.eval() # 모델을 평가모드로!
        eval_loss = 0

        results_pred = []
        results_real = []
        with torch.no_grad(): # evaluate()함수에는 단순 forward propagation만 할 뿐, gradient 계산 필요 X.
            for i, (label, text) in enumerate(data_loader):
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
        avg_eval_loss = eval_loss / len(data_loader.dataset)
        results_pred = np.array(results_pred)
        results_real = np.array(results_real)
        accuracy = np.sum(results_pred == results_real) / len(results_real)

        return avg_eval_loss, accuracy

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def objective(self, trial):
        # 하이퍼파라미터 탐색할 범위 지정
        size = trial.suggest_categorical('size', [50, 100])
        window = trial.suggest_categorical('window', [3, 5])
        min_count = trial.suggest_categorical('min_count', [1, 2])
        sg = trial.suggest_categorical('sg', [0, 1])

        # Word2Vec 모델 정의
        model = Word2Vec(size=size, window=window, min_count=min_count, sg=sg)

        # 모델 학습
        model.build_vocab(tokenized_corpus)
        model.train(tokenized_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        # 목적 함수(여기선 단순히 학습 손실값 반환) 설정
        loss = model.get_latest_training_loss()

        return loss


    def train_model(self, embedded_data, device, trn_loader, val_loader, VOCAB_SIZE, N_EPOCHS=10, LR=0.001, BATCH_SIZE=64):
        # 데이터셋 크기와 임베딩 차원 파악
        dataset_size, embedding_dim = embedded_data.shape
        vocab = set(word for sentence in tokenized_corpus for word in sentence)
        VOCAB_SIZE = len(vocab)
        # 모델 인스턴스 생성
        model = MyLSTM(input_dim=embedding_dim, hidden_dim=50, vocab_size=VOCAB_SIZE, n_classes=10)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss(reduction='sum')

        best_val_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_time = time.time()

            trn_loss = self.train(model=model,
                                  data_loader=trn_loader,
                                  criterion=loss_func,
                                  optimizer=optimizer,
                                  device=device)

            val_loss, accuracy = self.evaluate(model=model,
                                               data_loader=val_loader,
                                               criterion=loss_func,
                                               optimizer=optimizer,
                                               device=device)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {trn_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {100 * accuracy:.3f}%')

if __name__ == "__main__":
    # 데이터 전처리 및 모델 학습
    # 데이터 로드 등의 전처리

    # ... 데이터 로드 및 전처리 로직 ...

    # Word2Vec 모델 학습
    # word2vec_model = Word2Vec(...) # 이 부분은 이전 코드에서 학습한 모델을 가져와서 사용

    # embedded_data = torch.tensor(...) # 이 부분은 이전 코드에서 생성한 embedded_data를 가져와서 사용

    # 모델 인스턴스 생성
    text_model = TextModel()

    # 학습
    text_model.train_model(embedded_data)
