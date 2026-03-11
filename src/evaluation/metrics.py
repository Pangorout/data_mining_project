import numpy as np
from sklearn.metrics import f1_score, classification_report, mean_absolute_error, mean_squared_error

def evaluate_classification(y_true, y_pred):
    """Tính toán các metric đánh giá cho mô hình Phân lớp."""
    f1_macro = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred)
    
    print(f"F1-macro Score: {f1_macro:.4f}")
    print("\nChi tiết Classification Report:")
    print(report)
    
    return f1_macro

def evaluate_regression(y_true, y_pred):
    """Tính toán các metric đánh giá cho mô hình Hồi quy."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"MAE (Sai số tuyệt đối trung bình): {mae:.4f}")
    print(f"RMSE (Căn bậc hai sai số toàn phương trung bình): {rmse:.4f}")
    
    return mae, rmse