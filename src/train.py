# src/train.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import argparse
import os


def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and ensures it has the required columns.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame with at least 'text' and 'label' columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found at the given path.
    ValueError
        If the CSV cannot be parsed or is missing required columns.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        # Provide a clearer message about the missing file
        raise FileNotFoundError(f"Data file not found at: {data_path}") from e
    except Exception as e:
        # Catch other read/parsing errors (e.g. malformed CSV)
        raise ValueError(f"Failed to read CSV from {data_path}: {e}") from e

    # Validate required columns
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, "
            f"but only these were found: {set(df.columns)}"
        )
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the DataFrame into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset with 'text' and 'label' columns.

    Returns
    -------
    (X_train, X_test, y_train, y_test) : tuple of pd.Series
        Train/test split for features and labels.
    """
    try:
        # Prefer a stratified split so label proportions are similar
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )
    except ValueError:
        # Fallback if stratification fails (e.g., very small dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            random_state=42,
        )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """
    Builds and trains a text classification pipeline.

    The pipeline uses:
      - TfidfVectorizer: to convert text into numerical features
      - LogisticRegression: as the classifier

    Parameters
    ----------
    X_train : pd.Series
        Training texts.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    Pipeline
        A fitted scikit-learn Pipeline ready for evaluation or saving.
    """
    clf_pipeline = make_pipeline(
        TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    clf_pipeline.fit(X_train, y_train)
    return clf_pipeline


def save_model(model: Pipeline, model_path: str) -> None:
    """
    Saves the trained model to a file using joblib.

    Parameters
    ----------
    model : Pipeline
        The trained model pipeline.
    model_path : str
        Path where the model should be stored (e.g. 'models/model.joblib').
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)
    print(f"Saved model to {model_path}")


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Prints a simple summary of the dataset.

    This is a small helper function to give an overview of what we are
    training on, which can be useful for debugging and EDA.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset with at least 'text' and 'label' columns.
    """
    print("===== Dataset Summary =====")
    print(f"Number of rows: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())
    print("===========================")


def main(data_path: str, model_path: str) -> None:
    """
    Main workflow to load data, print a summary, train, evaluate,
    and save the model.

    Parameters
    ----------
    data_path : str
        Path to the training CSV file.
    model_path : str
        Path where the trained model will be saved.
    """
    # Load and validate the dataset (with error handling inside the function)
    df = load_and_validate_data(data_path)

    # New feature: print a brief dataset summary
    summarize_dataset(df)

    # Split, train, and evaluate
    X_train, X_test, y_train, y_test = split_data(df)
    clf = train_model(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

    # Save the trained pipeline to disk
    save_model(clf, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument(
        "--data",
        default="data/sentiments.csv",
        help="Path to the CSV file containing 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--out",
        default="models/sentiment.joblib",
        help="Output path for the trained model file.",
    )

    args: argparse.Namespace = parser.parse_args()
    main(data_path=args.data, model_path=args.out)
