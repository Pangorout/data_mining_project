import pandas as pd
import yaml
import os

# Đọc cấu hình từ file params.yaml
def load_params(config_path="configs/params.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# Tải dữ liệu thô
def load_raw_data(config_path="configs/params.yaml"):
    params = load_params(config_path)
    data_path = params["paths"]["raw_data"]
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file tại {data_path}. Vui lòng kiểm tra lại!")
        
    df = pd.read_csv(data_path)
    print(f"Đã tải thành công dataset với {df.shape[0]} dòng và {df.shape[1]} cột.")
    return df