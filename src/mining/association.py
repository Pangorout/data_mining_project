import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def perform_apriori(df, rating_col='reviews.rating', min_support=0.05, min_confidence=0.2):
    """Tìm luật kết hợp giữa các khía cạnh (aspects) và cảm xúc (khen/phàn nàn)."""
    print("Đang chuẩn bị dữ liệu giỏ hàng cho Apriori...")
    
    # Tạo nhãn Phàn nàn (Negative) hoặc Khen (Positive) dựa trên Rating
    # Giả sử rating từ 1-3 là Phàn nàn (0), 4-5 là Khen (1)
    df['is_Positive'] = df[rating_col].apply(lambda x: 1 if float(x) >= 4 else 0)
    df['is_Negative'] = df[rating_col].apply(lambda x: 1 if float(x) < 4 else 0)
    
    # Lấy ra các cột dạng boolean (0/1) đã tạo ở bước Feature Engineering
    # Bao gồm các cột 'has_...' và 2 cột cảm xúc vừa tạo
    basket_cols = [col for col in df.columns if col.startswith('has_')] + ['is_Positive', 'is_Negative']
    basket = df[basket_cols].astype(bool)
    
    print(f"Bắt đầu chạy thuật toán Apriori với min_support={min_support}...")
    # 1. Tìm tập phổ biến (Frequent Itemsets)
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("Không tìm thấy tập phổ biến nào với min_support hiện tại. Hãy thử giảm min_support xuống!")
        return None
        
    # 2. Sinh luật kết hợp (Association Rules)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Lọc ra các luật có chứa 'is_Negative' hoặc 'is_Positive' ở hệ quả (consequents)
    # Để xem khía cạnh nào (room, staff,...) dẫn đến lời khen hay tiếng chê
    def is_sentiment_rule(consequents):
        return 'is_Positive' in consequents or 'is_Negative' in consequents

    sentiment_rules = rules[rules['consequents'].apply(is_sentiment_rule)]
    
    # Sắp xếp theo độ tin cậy (confidence) và lift giảm dần
    sentiment_rules = sentiment_rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    
    print(f"Đã tìm thấy {len(sentiment_rules)} luật liên quan đến Khen/Phàn nàn.")
    return sentiment_rules