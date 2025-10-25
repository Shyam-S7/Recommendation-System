import pickle
import os


def save_pkl(obj, filename):
    """Save any Python object as a pickle file."""
    os.makedirs("artifacts", exist_ok=True)
    path = os.path.join("artifacts", filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"✅ Saved: {path}")
