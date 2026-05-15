# Text Preprocessing
import re
import string
import joblib
import pandas as pd

def clean_text(text):
  if pd.isna(text):
    return ""
  
  text = str(text)
  text = text.lower()
  
  text = re.sub(r"http\S+|www\S+", " ", text)
  text = re.sub(r"<.*?>", " ", text)
  text = re.sub(r"\d+", " ", text)
  
  text = text.translate(str.maketrans("", "", string.punctuation))
  
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text

def preprocess_dataframe(df, text_column="comment_text"):
  df = df.copy()
  df["cleaned_text"] = df[text_column].apply(clean_text)
  
  before = len(df)
  df = df[df["cleaned_text"] != ""]
  after = len(df)
  print("Removed empty rows:", before - after)
  return df.reset_index(drop=True)

def show_examples(df, text_column="comment_text", n=5):
  sample = df.head(n)
  for i, row in sample.iterrows():
    print("Before:", row[text_column])
    print("After:", clean_text(row[text_column]))
    print("-" * 40)
    
def save_preprocessor(path="models/preprocessor_config.joblib"):
  config = {
    "text_column": "comment_text",
    "cleaned_column": "cleaned_text",
    "steps": [
      "lowercase",
      "remove_urls",
      "remove_html",
      "remove_numbers",
      "remove_punctuation",
      "remove_extra_spaces"
    ]
  }
  
  joblib.dump(config, path)
  print("Preprocessor config saved.")
  
def load_preprocessor(path="models/preprocessor_config.joblib"):
  config = joblib.load(path)
  print("Preprocessor config loaded.")
  return config