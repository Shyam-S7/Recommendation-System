# from data_ingestion import load_data
import ast
import re
import nltk
from logger import logging
from exception import CustomException
import pandas as pd

# nltk.download("wordnet")
# nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def clean_data():
    df = pd.read_csv(
        csv_path,
        usecols=["brand", "colour", "description", "p_attributes", "name", "img"],
    )
    # df = load_data()
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    for col in [
        "name",
        "brand",
        "description",
        "p_attributes",
        "img",
        "colour",
    ]:
        df[col] = df[col].astype(str).str.strip()

    html_tags = ["<br>", "<b>", "</li>", "</ul>", "</b>", "<li>", "<ul>"]
    for tag in html_tags:
        df["description"] = df["description"].str.replace(tag, "", regex=True)

    df["description"] = (
        df["description"].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.strip()
    )

    df["p_attributes"] = df["p_attributes"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["attributes"] = df["p_attributes"].apply(
        lambda x: " ".join(x.values()) if isinstance(x, dict) else ""
    )
    df["attributes"] = (
        df["attributes"].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.strip()
    )
    df.drop(columns="p_attributes", inplace=True)

    brand = df["brand"].unique()
    color = df["colour"].unique()
    gender = ["women", "men", "boys", "girls"]
    remove_list = list(brand) + list(color) + gender

    for word in remove_list:
        df["name"] = df["name"].str.replace(
            r"\b{}\b".format(re.escape(word)), "", regex=True, flags=re.IGNORECASE
        )

    df["name"] = df["name"].str.replace(r"[^a-zA-Z\s]", "", regex=True).str.strip()

    df["tags"] = df["brand"] + " " + df["description"] + " " + df["attributes"]
    df.drop(columns=["brand", "colour", "description", "attributes"], inplace=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def process_tags(tags):
        words = tags.lower().split()
        seen = set()
        clean_words = []
        for word in words:
            if word in stop_words:
                continue
            word = lemmatizer.lemmatize(word)
            if word not in seen:
                seen.add(word)
                clean_words.append(word)
        return clean_words

    df["tags"] = df["tags"].apply(process_tags)
    logging.info(" Data cleaning completed successfully!")
    return df


cleaned_df = clean_data()
print(cleaned_df.head())
cleaned_df.to_csv("preprocessed.csv", index=False)

logging.info(f"Cleaned data: {cleaned_df.columns}")
logging.info(f"Cleaned data: {cleaned_df['tags']}")

# if __name__ == "__main__":
# python src\data_preprocessing.py
