# tests/test_predict.py

import os
import sys
import pytest

# Ensure the project root is on sys.path so we can import src.predict
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_model, predict_texts

MODEL_PATH = "models/sentiment.joblib"


@pytest.fixture(scope="session")
def classifier():
    """
    Load the trained classifier once per test session.
    Fails early if the model file is missing.
    """
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    return load_model(MODEL_PATH)


@pytest.mark.parametrize(
    "text",
    [
        "I love this movie, it was fantastic and inspiring!",
        "The service was terrible and the food was awful.",
    ],
)
def test_model_returns_valid_label_and_prob(classifier, text):
    """
    Sanity-check: for any input text, the model returns:
      - exactly one prediction
      - exactly one probability (or None if not supported)
      - a label in the expected set {0, 1}
    This confirms the prediction pipeline loads and runs end-to-end.
    """
    preds, probs = predict_texts(classifier, [text])

    # One prediction and one probability per input text
    assert len(preds) == 1
    assert len(probs) == 1

    # Label is one of the known classes
    assert preds[0] in (0, 1)

    # If your classifier supports predict_proba, prob should be a float
    # (if not, your predict_texts already returns None for each prob)
    # This assertion is mild; it just ensures the type is consistent.
    assert probs[0] is None or isinstance(probs[0], float)
