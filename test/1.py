def test_data_file_exists():
    import os

    assert os.path.exists("data/train.csv"), "❌ train.csv missing!"
