import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_evaluate_regressor(X_train, X_test, y_train, y_test, model_type='ridge'):
    """Huấn luyện và đánh giá mô hình hồi quy dự đoán điểm Rating."""
    if model_type == 'ridge':
        reg = Ridge(random_state=42) # Baseline
    elif model_type == 'rf':
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4) # Mô hình mạnh
    else:
        raise ValueError("Model type không hợp lệ!")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000, stop_words='english')),
        ('reg', reg)
    ])

    print(f"Đang huấn luyện mô hình hồi quy {model_type.upper()}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Đánh giá bằng MAE và RMSE theo yêu cầu
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    
    import joblib
    import os
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(pipeline, f"outputs/models/regressor_{model_type}.pkl")
    
    return pipeline, y_pred, mae, rmse