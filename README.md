# ChatbotLangChain

Dự án **ChatbotLangChain** là một chatbot sử dụng **LangChain** để:
- Trả lời câu hỏi dựa trên dữ liệu do người dùng **upload**.
- Thực hiện **tính toán hồi quy tuyến tính** từ dữ liệu được cung cấp.

---

## 1. Yêu cầu hệ thống
- Python **3.13**
- **pip** (Python package manager)

---

## 2. Cài đặt môi trường ảo (tuỳ chọn)
Khuyến nghị tạo môi trường ảo để tránh xung đột thư viện:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:
- **Windows**:
```bash
venv\Scripts\activate
```
- **MacOS / Linux**:
```bash
source venv/bin/activate
```

---

## 3. Cài đặt thư viện từ `requirements.txt`
Sau khi clone dự án, chạy lệnh sau để cài tất cả thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

## 4. Chạy dự án
Chạy file chính:
```bash
streamlit run app.py
```

---

## 5. Cấu trúc dự án (ví dụ)
```
ChatbotLangChain/
│-- app.py               # File chạy chính
│-- requirements.txt     # File chứa danh sách thư viện
│-- data/                # Thư mục chứa dữ liệu (nếu có)
│-- README.md            # File mô tả dự án
```

---

## 6. Chức năng chính
- **Upload dữ liệu**: Người dùng có thể upload file dữ liệu (CSV, TXT,...).
- **Chatbot trả lời**: Bot sử dụng LangChain để xử lý dữ liệu và trả lời câu hỏi liên quan.
- **Hồi quy tuyến tính**: Bot có thể thực hiện tính toán hồi quy tuyến tính từ dữ liệu.
