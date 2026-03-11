import papermill as pm
import os

os.makedirs("outputs/reports", exist_ok=True)

# Danh sách luồng pipeline theo đúng thứ tự
notebooks = [
    "01_eda.ipynb",
    "02_preprocess_feature.ipynb",
    "03_mining_or_clustering.ipynb",
    "04_modeling.ipynb",
    "04b_semi_supervised.ipynb",
    "05_evaluation_report.ipynb"
]

print("Bắt đầu chạy pipeline")
print("-" * 50)

for nb in notebooks:
    input_path = f"notebooks/{nb}"
    output_path = f"outputs/reports/executed_{nb}" 
    
    print(f"⏳ Đang thực thi: {nb}...")
    try:
        # Papermill sẽ mở notebook, chạy từng cell và lưu kết quả lại
        pm.execute_notebook(
            input_path,
            output_path,
            kernel_name='python3'
        )
        print(f"Hoàn tất: {nb}\n")
    except Exception as e:
        print(f"Lỗi khi chạy {nb}. Dừng pipeline")
        print(e)
        break

print("-" * 50)
print("PIPELINE ĐÃ CHẠY XONG! Vui lòng kiểm tra các thư mục:")
print("- Ảnh biểu đồ: outputs/figures/")
print("- Mô hình: outputs/models/")
print("- Notebook đã chạy: outputs/reports/")