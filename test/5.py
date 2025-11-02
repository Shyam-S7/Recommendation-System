def test_model_file_saved():
    import os
    from src.train import train_model

    train_model()
    assert os.path.exists("models/model.pkl"), "❌ Model file NOT saved!"
