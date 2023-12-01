from gensim.models import Word2Vec
import optuna
from sklearn.model_selection import train_test_split


class Embedding:
    def __init__(self, args, tokenized_sentences):
        self.args = args
        self.corpus = tokenized_sentences
        self.trn, self.tst, self.val = self.get_split_data()
        # self.get_embedding_vector()

    # TODO: corpus를 train, test, validation으로 split 하여 임베딩 진행 or 임베딩 진행 후 split -> 나중에 argparser로 선택할 수 있도록 구현
    def get_split_data(self):
        if self.args.sp_option == 0:
            # 옵션 0: 나누지 않고 전체 코퍼스를 훈련에 사용
            train_corpus = self.corpus
            test_corpus = None
            validation_corpus = None
        else:
            # 옵션 1: 코퍼스를 훈련, 테스트, 검증 세트로 나눔
            train_corpus, temp_corpus = train_test_split(
                self.corpus, test_size=0.1, random_state=42
            )
            test_corpus, validation_corpus = train_test_split(
                temp_corpus, test_size=0.4, random_state=42
            )
        return train_corpus, test_corpus, validation_corpus

    def objective(self, trial):
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
        vectors = []
        for i in [self.trn, self.tst, self.val]:
            if i is not None:
                best_model.build_vocab(i, update=False)
                best_model.train(
                    self.corpus,
                    total_examples=best_model.corpus_count,
                    epochs=best_model.epochs,
                )
                word_vectors = best_model.wv
                vectors.append(word_vectors)
        print("형태 : ", word_vectors.vectors.shape)
        print("데이터 : ", i)
        
        # 3개의 벡터를 return
        train_corpus, temp_corpus = train_test_split(
                self.corpus, test_size=0.5, random_state=42
            )
        test_corpus, validation_corpus = train_test_split(
                temp_corpus, test_size=0.5, random_state=42
            )
        return train_corpus, test_corpus, validation_corpus


if __name__ == "__main__":
    # 임베딩 옵션 설정

    embedding_option = 0
    embedding = Embedding("../tokenized_0.csv")
    train_data, test_data, validation_data = embedding.get_split_data(embedding_option)
    word_vectors = embedding.get_embedding_vector()
    # train, test, validation split에서 문제점이 있음 디버깅 필요
    # 전체를 대상으로 하이퍼 파라미터를 튜닝 한 뒤, TRAIN, TEST, VALIDATION에 대해서 하이퍼 파라미터 학습시켜서 임베딩 벡터를 각각 만들어냄.