import pandas as pd
import re
import ast

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return " ".join(text.split())

def load_analytics_data(file_path):
    print("⏳ [Data Ingestion] Reading 186MB raw records array...")
    df = pd.read_csv(file_path)
    
    df['title'] = df['title'].fillna('')
    df['summary'] = df['summary'].fillna('')
    df['category_code'] = df['category_code'].fillna('unknown')
    df['first_author'] = df['first_author'].fillna('Unknown')
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df['publish_year'] = df['published_date'].dt.year
    
    print("🧹 [Data Ingestion] Standardizing unstructured features...")
    df['cleaned_text'] = (df['title'] + " " + df['summary']).apply(clean_text)
    
    def parse_authors(val):
        try:
            return ast.literal_eval(val) if isinstance(val, str) and val.startswith('[') else [val]
        except:
            return [str(val)]
            
    df['parsed_authors'] = df['authors'].apply(parse_authors)
    return df.dropna(subset=['publish_year'])