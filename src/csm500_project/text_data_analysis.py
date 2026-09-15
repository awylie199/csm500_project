"""Text Data Analysis"""

from typing import Any, Final

import pandas as pd
from numpy import dtype, ndarray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Apply a very high frequency due to the high number of infrequent terms to minimise dimensionality
_MINIMUM_DATA_FREQUENCY: Final[int] = 1000
_MAX_DATA_FREQUENCY: Final[float] = 0.8
_TEXT_FIELDS: Final[tuple[str, ...]] = ("Description",)


def text_data_analysis(x_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Textual Data Analysis and Preprocessing"""

    count_vectorizer_transformer: CountVectorizer = CountVectorizer(
        min_df=_MINIMUM_DATA_FREQUENCY, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_vectorizer_transformer: TfidfVectorizer = TfidfVectorizer(
        min_df=_MINIMUM_DATA_FREQUENCY, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    sentence_transformer: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

    count_vectorizer_doc_terms = count_vectorizer_transformer.fit_transform(x_train[_TEXT_FIELDS[0]]).todense()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    tfidf_vectorizer_doc_terms = tfidf_vectorizer_transformer.fit_transform(x_train[_TEXT_FIELDS[0]]).todense()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    sbert_embeddings: ndarray[tuple[Any, ...], dtype[Any]] = sentence_transformer.encode(
        x_train[_TEXT_FIELDS[0]].tolist(), normalize_embeddings=True, batch_size=64, device="cuda"
    )

    count_vectorizer_df: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(count_vectorizer_doc_terms, columns=count_vectorizer_transformer.get_feature_names_out(), index=x_train.index),
            y_train,
        ],
        axis=1,
    )
    tfidf_vectorizer_df: pd.DataFrame = pd.concat(
        [
            pd.DataFrame(tfidf_vectorizer_doc_terms, columns=tfidf_vectorizer_transformer.get_feature_names_out(), index=x_train.index),
            y_train,
        ],
        axis=1,
    )

    sbert_df: pd.DataFrame = pd.concat(
        [pd.DataFrame(sbert_embeddings, columns=[f"sbert_{i}" for i in range(sbert_embeddings.shape[1])], index=x_train.index), y_train],
        axis=1,
    )

    print(f"Count Vectorizer Shape: {count_vectorizer_df.shape}")
    print(f"TF-IDF Vectorizer Shape: {tfidf_vectorizer_df.shape}")
    print(f"sBERT Embeddings Shape: {sbert_df.shape}")

    print(
        f"Vectorizer Vocabulary Difference: {len(set(count_vectorizer_transformer.vocabulary_.keys()).symmetric_difference(set(tfidf_vectorizer_transformer.vocabulary_.keys())))}"
    )

    return count_vectorizer_df, tfidf_vectorizer_df, sbert_df
