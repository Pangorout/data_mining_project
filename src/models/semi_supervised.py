import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def evaluate_semi_supervised(df, text_col='Cleaned_Review', label_col='Sentiment', label_rates=[0.1, 0.2, 0.3]):
    """Thực nghiệm Bán giám sát với các tỷ lệ nhãn khác nhau."""
    
    # Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col])
    
    results = []
    
    # Vector hóa TF-IDF chung
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train.fillna(''))
    X_test_vec = vectorizer.transform(X_test.fillna(''))
    
    # Lặp qua từng kịch bản tỷ lệ nhãn (10%, 20%, 30%)
    for rate in label_rates:
        rng = np.random.RandomState(42)
        # Tạo mask: True là bị che nhãn (unlabeled), False là giữ lại nhãn
        random_unlabeled_points = rng.rand(len(y_train)) > rate
        
        y_train_masked = np.copy(y_train)
        y_train_masked[random_unlabeled_points] = -1 # -1 đại diện cho dữ liệu không có nhãn
        
        # 1. Baseline: Chỉ dùng Supervised (Huấn luyện MỘT PHẦN dữ liệu có nhãn)
        X_train_labeled = X_train_vec[~random_unlabeled_points]
        y_train_labeled = y_train_masked[~random_unlabeled_points]
        
        base_clf = MultinomialNB()
        base_clf.fit(X_train_labeled, y_train_labeled)
        f1_base = f1_score(y_test, base_clf.predict(X_test_vec), average='macro')
        
        # 2. Semi-supervised: Dùng Self-Training (Học trên CẢ dữ liệu có nhãn và không nhãn)
        # Sử dụng base_clf làm mô hình nền tảng, ngưỡng tin cậy 0.8
        self_training_clf = SelfTrainingClassifier(MultinomialNB(), threshold=0.8)
        self_training_clf.fit(X_train_vec, y_train_masked)
        f1_semi = f1_score(y_test, self_training_clf.predict(X_test_vec), average='macro')
        
        results.append({
            'Label_Ratio': rate,
            'Supervised_F1': f1_base,
            'SemiSupervised_F1': f1_semi
        })
        
        print(f"Tỷ lệ nhãn {rate*100:.0f}% | Chỉ dùng Supervised F1: {f1_base:.4f} | Semi-supervised F1: {f1_semi:.4f}")
        
    return pd.DataFrame(results)