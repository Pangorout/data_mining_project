import pandas as pd

# Tập hợp các khía cạnh (aspects) phổ biến trong khách sạn
HOTEL_ASPECTS = ['room', 'staff', 'service', 'food', 'breakfast', 'location', 'clean', 'bed', 'bathroom', 'price']

def extract_aspects(text):
    """Tìm các khía cạnh dịch vụ xuất hiện trong review."""
    if not isinstance(text, str):
        return []
    words = set(text.split())
    # Tìm điểm giao thoa giữa từ trong review và danh sách aspect
    return [aspect for aspect in HOTEL_ASPECTS if aspect in words]

def build_features(df, text_col='Cleaned_Review'):
    """Tạo đặc trưng aspect và rời rạc hóa dữ liệu."""
    print("Đang trích xuất từ khóa khía cạnh (aspects)...")
    
    # Tạo cột dạng list chứa các aspect (giống dạng 'giỏ hàng')
    df['Aspects'] = df[text_col].apply(extract_aspects)
    
    # Rời rạc hóa (One-hot encoding) cho từng aspect để chạy Apriori/Classification
    for aspect in HOTEL_ASPECTS:
        df[f'has_{aspect}'] = df['Aspects'].apply(lambda x: 1 if aspect in x else 0)
        
    # Rời rạc hóa thêm cột độ dài review
    if 'Review_Length' in df.columns:
        df['Length_Bin'] = pd.cut(df['Review_Length'], 
                                  bins=[0, 20, 50, float('inf')], 
                                  labels=['Short', 'Medium', 'Long'])
        
    print(f"Đã tạo xong {len(HOTEL_ASPECTS)} đặc trưng khía cạnh.")
    return df