from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def train_evaluate_classifier(X_train, X_test, y_train, y_test, model_type='nb'):
    """Huấn luyện và đánh giá mô hình phân lớp."""
    # Khởi tạo mô hình dựa trên tham số
    if model_type == 'nb':
        clf = MultinomialNB() # Baseline 1
    elif model_type == 'logreg':
        clf = LogisticRegression(max_iter=1000, random_state=42) # Baseline 2
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4) # Mô hình cải tiến
    else:
        raise ValueError("Model type không hợp lệ!")

    # Tạo Pipeline: Chuyển text thành số -> Đưa vào mô hình
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english')),
        ('clf', clf)
    ])

    print(f"Đang huấn luyện mô hình {model_type.upper()}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Đánh giá bằng F1-macro theo đúng yêu cầu đề tài
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"F1-macro Score: {f1_macro:.4f}")
    
    import joblib
    import os
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(pipeline, f"outputs/models/classifier_{model_type}.pkl")
    
    return pipeline, y_pred, f1_macro