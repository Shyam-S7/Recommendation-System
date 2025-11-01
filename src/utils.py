import pickle
import os


def save_pkl(obj, filename):
    artifacts_path = os.path.join("data", "artifacts")  # ✅ "data/artifacts"
    os.makedirs(artifacts_path, exist_ok=True)  # ✅ creates folder if missing

    full_path = os.path.join(artifacts_path, filename)
    with open(full_path, "wb") as f:
        pickle.dump(obj, f)

    print(f"✅ Saved: {full_path}")
