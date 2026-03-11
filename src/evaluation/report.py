import pandas as pd

def generate_error_analysis(X_test, y_true, y_pred):
    """
    Tổng hợp dữ liệu và trích xuất các ca dự đoán sai (False Positives & False Negatives).
    Phục vụ trực tiếp cho phần Thảo luận và Insight trong báo cáo.
    """
    df_results = pd.DataFrame({
        'Review_Text': X_test,
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Lọc ra những dòng máy đoán sai
    errors = df_results[df_results['Actual'] != df_results['Predicted']]
    
    # 0 là Phàn nàn (Negative), 1 là Khen (Positive)
    false_positives = errors[(errors['Actual'] == 0) & (errors['Predicted'] == 1)]
    false_negatives = errors[(errors['Actual'] == 1) & (errors['Predicted'] == 0)]
    
    print(f"Tổng số ca dự đoán sai: {len(errors)}")
    print(f"- False Positives (Thực tế CHÊ, máy đoán KHEN): {len(false_positives)} ca")
    print(f"- False Negatives (Thực tế KHEN, máy đoán CHÊ): {len(false_negatives)} ca")
    
    return false_positives, false_negatives