import pandas as pd
from gensim.models import Word2Vec
import optuna


class Embedding:
    def __init__(self, corpus):
        self.df = pd.read_csv(corpus)
        # 텍스트를 토큰화하여 리스트로 변환
        self.corpus = [
            str(sentence).lower().split()
            for sentence in self.df["comments"]
            if pd.notnull(sentence)
        ]

    def objective(self, trial):
        # 하이퍼파라미터 탐색할 범위 지정
        vector_size = trial.suggest_int("vector_size", 10, 300)
        window = trial.suggest_int("window", 3, 10)
        min_count = trial.suggest_int("min_count", 5, 30)
        sg = trial.suggest_categorical("sg", [0])
        # Word2Vec 모델 정의
        model = Word2Vec(
            vector_size=vector_size, window=window, min_count=min_count, sg=sg
        )
        # 모델 학습
        # print(tokenized_corpus)
        model.build_vocab(self.corpus)
        model.train(
            corpus_iterable=self.corpus,
            total_examples=model.corpus_count,
            epochs=model.epochs,
            compute_loss=True,
        )
        # 목적 함수(여기선 단순히 학습 손실값 반환) 설정
        loss = model.get_latest_training_loss()
        return loss

    def get_embedding_vector(self):
        # Optuna를 사용하여 최적화 실행
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=100)
        # 최적의 하이퍼파라미터 출력
        print("Best parameters:", study.best_params)
        # 최적의 하이퍼파라미터로 모델 재훈련
        best_params = study.best_params
        best_model = Word2Vec(
            vector_size=best_params["vector_size"],
            window=best_params["window"],
            min_count=best_params["min_count"],
            sg=best_params["sg"],
        )
        best_model.build_vocab(self.corpus)
        best_model.train(
            self.corpus,
            total_examples=best_model.corpus_count,
            epochs=best_model.epochs,
        )
        # 튜닝된 Word2Vec 모델에서 단어 벡터 출력
        word_vectors = best_model.wv
        print("형태: ", word_vectors.vectors.shape)
        return word_vectors


if __name__ == "__main__":
    embedding = Embedding("./tokenized_0.csv")
    word_vectors = embedding.get_embedding_vector()
