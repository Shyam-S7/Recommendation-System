from src.preprocessing import preprocess


def test_preprocess_shape():
    import pandas as pd

    df = pd.read_csv("data/train.csv")
    processed = preprocess(df)
    assert processed.shape[0] > 0, "❌ Preprocessing removed all rows!"
