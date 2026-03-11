import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def perform_topic_clustering(df, text_col='Cleaned_Review', n_clusters=5, max_features=1000):
    """
    Thực hiện TF-IDF và K-Means clustering để tìm chủ đề của review.
    """
    print(f"Bắt đầu Vector hóa TF-IDF với tối đa {max_features} đặc trưng...")
    
    # 1. Vector hóa văn bản
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = tfidf.fit_transform(df[text_col].fillna(''))
    
    # 2. Phân cụm K-Means
    print(f"Đang chạy K-Means với số cụm K = {n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_tfidf)
    
    # 3. Đánh giá chất lượng cụm bằng Silhouette Score
    # Để tránh tràn RAM máy tính, ta chỉ lấy mẫu ngẫu nhiên 5000 dòng để tính điểm
    sample_size = min(5000, X_tfidf.shape[0])
    sil_score = silhouette_score(X_tfidf, df['Cluster'], sample_size=sample_size, random_state=42)
    print(f"Điểm Silhouette Score: {sil_score:.4f}")
    
    # 4. Đặt tên cụm dựa trên Top từ khóa
    terms = tfidf.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    cluster_names = {}
    print("\n--- PHÂN TÍCH CHỦ ĐỀ CÁC CỤM ---")
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :5]]
        cluster_name = " - ".join(top_terms)
        cluster_names[i] = cluster_name
        print(f"Cụm {i}: {cluster_name}")
        
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    
    return df, kmeans, tfidf, sil_score

def get_representative_reviews(df, cluster_col='Cluster_Name', review_col='reviews.text', top_n=2):
    """Lấy ra các review đại diện cho mỗi cụm."""
    representatives = {}
    for cluster in df[cluster_col].unique():
        # Lấy ngẫu nhiên top_n review dài vừa phải để làm đại diện
        sample_reviews = df[(df[cluster_col] == cluster) & (df['Review_Length'] > 5)].sample(min(top_n, df.shape[0]), random_state=42)
        representatives[cluster] = sample_reviews[review_col].tolist()
    return representatives