from pathlib import Path
import re
import ast
from collections import Counter

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "goodreads_top100_from1980to2023_final.csv"


def parse_genres(value):
    """
    Converts the genres column from text format into a real Python list.

    Example:
    "['Fiction', 'Fantasy', 'Romance']"
    becomes:
    ['Fiction', 'Fantasy', 'Romance']
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed_value = ast.literal_eval(value)

        if isinstance(parsed_value, list):
            return [str(genre).strip() for genre in parsed_value]

    except Exception:
        pass

    return [genre.strip() for genre in str(value).split(",") if genre.strip()]


def load_and_prepare_data(test_size=0.20, random_state=42):
    """
    Loads the Goodreads dataset, cleans it, creates the target variable,
    processes genre features, and returns train-test data for model training.
    """

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    numeric_columns = [
        "num_pages",
        "rating_score",
        "num_ratings",
        "num_reviews",
        "current_readers",
        "want_to_read",
        "price"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["current_readers"] = df["current_readers"].fillna(0)
    df["want_to_read"] = df["want_to_read"].fillna(0)

    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["publication_year"] = df["publication_date"].dt.year
    df["publication_month"] = df["publication_date"].dt.month

    df["is_series"] = (
        df["series_title"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    df["description"] = df["description"].fillna("")
    df["description_length"] = df["description"].apply(len)

    print("Creating target variable...")

    df["reader_interest"] = df["current_readers"] + df["want_to_read"]

    interest_threshold = df["reader_interest"].median()

    df["high_reader_interest"] = (
        df["reader_interest"] >= interest_threshold
    ).astype(int)

    print("Reader interest threshold:", interest_threshold)
    print("Target distribution:")
    print(df["high_reader_interest"].value_counts())

    print("Processing genre features...")

    df["genres_list"] = df["genres"].apply(parse_genres)

    genre_counter = Counter()

    for genre_list in df["genres_list"]:
        genre_counter.update(genre_list)

    top_genres = [genre for genre, count in genre_counter.most_common(20)]

    print("Top 20 genres selected:")
    for genre in top_genres:
        print("-", genre)

    genre_columns = []

    for genre in top_genres:
        clean_genre_name = re.sub(r"\W+", "_", genre.lower()).strip("_")
        column_name = "genre_" + clean_genre_name

        df[column_name] = df["genres_list"].apply(
            lambda genre_list, selected_genre=genre: 1 if selected_genre in genre_list else 0
        )

        genre_columns.append(column_name)

    numeric_features = [
        "rating_score",
        "num_pages",
        "price",
        "publication_year",
        "publication_month",
        "is_series",
        "description_length"
    ] + genre_columns

    categorical_features = [
        "language",
        "format"
    ]

    text_feature = "description"

    X = df[numeric_features + categorical_features + [text_feature]]
    y = df["high_reader_interest"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder)
        ]
    )

    text_transformer = TfidfVectorizer(
        max_features=1000,
        stop_words="english"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
            ("text", text_transformer, text_feature)
        ],
        sparse_threshold=0.0
    )

    metadata = {
        "dataset_shape": df.shape,
        "interest_threshold": interest_threshold,
        "top_genres": top_genres,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_feature": text_feature
    }

    return X_train, X_test, y_train, y_test, preprocessor, metadata


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, metadata = load_and_prepare_data()

    print("\nData preparation completed successfully.")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Metadata:")
    print(metadata)