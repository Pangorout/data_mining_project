import matplotlib.pyplot as plt
import seaborn as sns
import os

# Tự động tạo thư mục chứa ảnh nếu chưa có
os.makedirs("outputs/figures", exist_ok=True)

def plot_eda_distributions(df, rating_col='reviews.rating', length_col='Review_Length'):
    """Vẽ và lưu biểu đồ EDA (Notebook 01)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.countplot(data=df, x=rating_col, ax=axes[0], palette='viridis')
    axes[0].set_title('Phân phối điểm đánh giá')
    
    sns.histplot(data=df, x=length_col, bins=50, kde=True, ax=axes[1], color='coral')
    axes[1].set_title('Phân phối độ dài Review')
    
    plt.tight_layout()
    plt.savefig("outputs/figures/01_eda_distribution.png", dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix - Logistic Regression'):
    """Vẽ và lưu Ma trận nhầm lẫn (Notebook 04)."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.savefig("outputs/figures/04_confusion_matrix.png", dpi=300)
    plt.show()

def plot_learning_curve(results_df):
    """Vẽ và lưu Learning Curve Bán giám sát (Notebook 04b)."""
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['Label_Ratio'] * 100, results_df['Supervised_F1'], marker='o', label='Supervised Baseline', linestyle='--', color='red')
    plt.plot(results_df['Label_Ratio'] * 100, results_df['SemiSupervised_F1'], marker='s', label='Semi-supervised', linewidth=2, color='green')
    
    plt.title('Learning Curve: Đánh giá sức mạnh của Semi-supervised')
    plt.xlabel('Tỷ lệ dữ liệu có nhãn (%)')
    plt.ylabel('F1-Macro Score')
    plt.xticks([10, 20, 30])
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig("outputs/figures/04b_learning_curve.png", dpi=300)
    plt.show()