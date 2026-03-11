import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Tải bộ từ điển stopwords
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Làm sạch văn bản: chữ thường, bỏ dấu, số và stop words."""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    clean_words = [w for w in text.split() if w not in stop_words]
    
    return " ".join(clean_words)

def preprocess_dataframe(df, text_col='reviews.text', rating_col='reviews.rating'):
    """Làm sạch toàn bộ DataFrame và tạo đặc trưng mới."""
    print(f"Kích thước ban đầu: {df.shape}")
    
    # Bỏ các dòng thiếu text hoặc rating
    df = df.dropna(subset=[text_col, rating_col]).copy()
    
    print("Đang xử lý ngôn ngữ tự nhiên (NLP) cho review, vui lòng đợi...")
    df['Cleaned_Review'] = df[text_col].apply(clean_text)
    df['Review_Length'] = df['Cleaned_Review'].apply(lambda x: len(x.split()))
    
    # Loại bỏ các review sau khi clean bị rỗng (ví dụ: review chỉ chứa icon/số)
    df = df[df['Review_Length'] > 0]
    
    print(f"Kích thước sau khi làm sạch: {df.shape}")
    return df