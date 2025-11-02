import pandas as pd


def test_dataset_not_empty():
    df = pd.read_csv("data/train.csv")
    assert len(df) > 0, "❌ Dataset is EMPTY!"
