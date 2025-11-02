from src.model import load_model


def test_recommend_output():
    model = load_model()
    result = model.recommend(user_id=1)
    assert isinstance(result, list), "❌ recommend() must return a list"
    assert len(result) > 0, "❌ recommend() returned empty list"
