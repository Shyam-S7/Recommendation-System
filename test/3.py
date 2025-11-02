REQUIRED_COLS = ["user_id", "item_id", "rating"]


def test_required_columns():
    df = pd.read_csv("data/train.csv")
    for col in REQUIRED_COLS:
        assert col in df.columns, f"❌ Missing column: {col}"
