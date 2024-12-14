import pandas as pd
import re

def clean_text(text):
    
    text = text.lower()  
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(file_path):
    
    df = pd.read_csv(file_path)
    
    df = df[df['lang_abv'] == 'en']
    
    df['premise'] = df['premise'].apply(clean_text)
    df['hypothesis'] = df['hypothesis'].apply(clean_text)
    
    return df

def main():
    file_path = 'train.csv'
    cleaned_data = preprocess_data(file_path)
    cleaned_data.to_csv('train_cleaned.csv', index=False)
    
    print("Data preprocessing complete. Cleaned data saved to curr directory")

if __name__ == "__main__":
    main()
