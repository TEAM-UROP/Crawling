import pandas as pd
from gensim.models import Word2Vec
import optuna
from sklearn.model_selection import train_test_split

class Embedding:
    def __init__(self, corpus):
        self.df = pd.read_csv(corpus)
        # 텍스트를 토큰화하여 리스트로 변환
        self.corpus = [
            str(sentence).lower().split()
            for sentence in self.df["comments"]
            if pd.notnull(sentence)
        ]

    def get_split_data(self, option):
        if option == 0:
            # 옵션 0: 나누지 않고 전체 코퍼스를 훈련에 사용
            train_corpus = self.corpus
            test_corpus = None
            validation_corpus = None
        else:
            # 옵션 1: 코퍼스를 훈련, 테스트, 검증 세트로 나눔
            train_corpus, temp_corpus = train_test_split(self.corpus, test_size=0.5, random_state=42)
            test_corpus, validation_corpus = train_test_split(temp_corpus, test_size=0.4, random_state=42)
        return train_corpus, test_corpus, validation_corpus

    def objective(self, trial, corpus):
        # 하이퍼파라미터 탐색할 범위 지정
        vector_size = trial.suggest_int("vector_size", 10, 100)
        window = trial.suggest_int("window", 3, 10)
        min_count = trial.suggest_int("min_count", 5, 30)
        sg = trial.suggest_categorical("sg", [0])
        # Word2Vec 모델 정의
        model = Word2Vec(
            vector_size=vector_size, window=window, min_count=min_count, sg=sg
        )
        # 모델 학습
        model.build_vocab(corpus)
        model.train(
            corpus_iterable=corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs,
            compute_loss=True,
        )
        # 목적 함수(여기선 단순히 학습 손실값 반환) 설정
        loss = model.get_latest_training_loss()
        return loss, model  # 모델 반환 추가

    def get_embedding_vector(self, train_corpus, test_corpus, validation_corpus):
        # Optuna를 사용하여 최적화 실행
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, train_corpus), n_trials=100)
        # 최적의 하이퍼파라미터 출력
        print("Best parameters:", study.best_params)

        # 전체 데이터셋을 사용하여 모델 재훈련
        best_params = study.best_params
        best_model = Word2Vec(
            vector_size=best_params["vector_size"],
            window=best_params["window"],
            min_count=best_params["min_count"],
            sg=best_params["sg"],
        )
        best_model.build_vocab(train_corpus)
        best_model.train(
            train_corpus,
            total_examples=best_model.corpus_count,
            epochs=best_model.epochs,
        )

        # 튜닝된 Word2Vec 모델 반환
        return best_model

if __name__ == "__main__":
    # 임베딩 옵션 설정
    embedding_option = 1
    embedding = Embedding("../tokenized_0.csv")
    train_corpus, test_corpus, validation_corpus = embedding.get_split_data(embedding_option)

    # Word2Vec 모델 훈련
    word2vec_model = embedding.get_embedding_vector(train_corpus, test_corpus, validation_corpus)

    # 각 데이터셋 벡터화
    train_vectors = word2vec_model.wv[train_corpus]
    test_vectors = word2vec_model.wv[test_corpus]
    validation_vectors = word2vec_model.wv[validation_corpus]

    # 이후에 train_vectors, test_vectors, validation_vectors를 사용하여 모델을 학습하거나 평가할 수 있습니다.
