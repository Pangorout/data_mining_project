# Phân tích đánh giá khách sạn & chủ đề dịch vụ
**Học phần**: Khai phá dữ liệu
**Giảng viên hướng dẫn**: Th.S Lê Thị Thùy Trang
1. Giới thiệu dự án
Dự án này tập trung khai phá tri thức từ các văn bản đánh giá (reviews) của khách hàng về dịch vụ khách sạn. Bằng việc kết hợp các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và Khai phá dữ liệu (Data Mining), hệ thống thực hiện luồng công việc tự động:
- **Tiền xử lí văn bản**: Làm sạch, chuẩn hóa và vector hóa dữ liệu chữ bằng TF-IDF.
- **Khai phá mẫu & Phân cụm (Unsupervised Learning)**: Sử dụng K-Means để gom cụm chủ đề và Apriori để tìm các luật kết hợp dịch vụ thường đi kèm với đánh giá khen/chê.
- **Mô hình hóa (Supervised Learning)**: Xây dựng các mô hình phân lớp (dự đoán Sentiment) và hồi quy (dự đoán điểm Rating).
- **Bán giám sát (Semi-supervised Learning)**: Áp dụng kĩ thuật Self-Training để giải quyết bài toán thiếu nhãn dữ liệu.

2. Nguồn dữ liệu & Từ điển dữ liệu (Data Dictionary)
- **Nguồn dữ liệu**: Dữ liệu được lấy từ [Hotel reviews datasets](https://www.kaggle.com/datasets/datafiniti/hotel-reviews)

**Từ điển các cột quan trọng (Data Dictionary):**
- reviews.text: Nội dung chi tiết lời đánh giá của khách hàng (Biến độc lập chính cho NLP).
- reviews.rating: Điểm số khách hàng đánh giá (Biến mục tiêu cho Hồi quy, từ 1-5).
- name: Tên khách sạn được đánh giá.
- categories / primaryCategories: Phân loại dịch vụ (ví dụ: Hotel, Resort, Motel).
- Sentiment: Nhãn phân lớp được tạo lập tự động (1: Khen khi Rating >= 4, 0: Phàn nàn khi Rating < 4).

3. Hướng dẫn chạy dự án
- **Bước 1**: Cài đặt môi trường: Mở terminal và cài đặt các thư viện cần thiết: `pip install -r requirements.txt`
- **Bước 2**: Chuẩn bị dữ liệu: Tải tập dữ liệu từ link ở trên, đặt vào thư mục `data/raw/`.
- **Bước 3**: Chạy tự động bằng Papermill: Hệ thống hỗ trợ thực thi tự động toàn bộ luồng pipeline từ 01 đến 05 mà không cần mở tay từng file notebook. Mở terminal và chạy lệnh `python scripts/run_papermill.py`

4. Insight chính:
- **Tiền xử lý NLP**: Loại bỏ stop words, làm sạch ký tự và thiết lập không gian vector tối ưu.
- **Luật dịch vụ (Apriori)**: Phát hiện ra sự liên kết chặt chẽ giữa các phàn nàn về vệ sinh (`clean`) và thái độ nhân viên (`staff`).
- **Xử lý thiếu nhãn**: Áp dụng thành công thuật toán Bán giám sát (Self-Training), chứng minh được sức mạnh tận dụng dữ liệu khuyết so với phương pháp Supervised truyền thống.
- **Khuyến nghị hành động**: Từ kết quả phân tích lỗi (Error Analysis), nhóm đề xuất chiến lược tối ưu nhân sự dọn phòng và cải tiến hệ thống form đánh giá dịch vụ đa chiều.
