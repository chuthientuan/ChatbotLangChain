import os
import streamlit as st
import pandas as pd
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Thư viện mới & cập nhật cho Phân tích Dữ liệu ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor  # Sử dụng Gradient Descent
from sklearn.metrics import mean_squared_error, r2_score

# --- Cấu hình Vertex AI ---
PROJECT_ID = "vivid-pen-471404-t6"
LOCATION = "us-central1"

# Khởi tạo Vertex AI
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)


# --- Hàm xử lý cho Chatbot (giữ nguyên) ---
@st.cache_resource
def create_conversational_chain(_vector_store):
    llm = VertexAI(model_name="gemini-2.5-flash", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(),
        memory=memory
    )
    return chain


# --- Giao diện người dùng Streamlit ---
def main():
    st.set_page_config(page_title="🎓 Trợ lý AI Đa năng", page_icon="💡", layout="wide")
    st.title("💡 Trợ lý AI Đa năng")

    tab1, tab2 = st.tabs(["🤖 Chatbot Hỏi Đáp PDF", "📊 Phân Tích & Hồi quy Dữ liệu"])

    # --- Giao diện Tab 1: Chatbot (giữ nguyên) ---
    with tab1:
        # (Code của tab chatbot giữ nguyên như cũ)
        st.header("Hỏi đáp tài liệu PDF")
        st.write("Upload một file PDF, sau đó đặt câu hỏi về nội dung bên trong.")
        uploaded_pdf = st.file_uploader("Chọn file PDF của bạn", type="pdf", key="pdf_uploader")
        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "processed_file_name" not in st.session_state:
            st.session_state.processed_file_name = None
        if uploaded_pdf is not None and st.session_state.processed_file_name != uploaded_pdf.name:
            st.info(f"Phát hiện file mới: '{uploaded_pdf.name}'. Bắt đầu xử lý...")
            with st.spinner("Đang xử lý tài liệu PDF... 🧠"):
                try:
                    with open(uploaded_pdf.name, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())
                    loader = PyPDFLoader(uploaded_pdf.name)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.split_documents(documents)
                    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
                    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
                    st.session_state.conversation_chain = create_conversational_chain(vector_store)
                    st.session_state.chat_history = []
                    st.session_state.processed_file_name = uploaded_pdf.name
                    st.success("Tài liệu đã được xử lý! Bạn có thể bắt đầu hỏi.")
                except Exception as e:
                    st.error(f"Lỗi khi xử lý file: {e}")
                finally:
                    os.remove(uploaded_pdf.name)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]): st.markdown(message["content"])
        user_question = st.chat_input("Đặt câu hỏi về nội dung tài liệu...")
        if user_question:
            if st.session_state.conversation_chain:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                with st.spinner("Bot đang suy nghĩ... 🤔"):
                    response = st.session_state.conversation_chain({"question": user_question})
                    answer = response["chat_history"][-1].content
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"): st.markdown(answer)
            else:
                st.warning("Vui lòng upload một file PDF trước.")

    # --- Giao diện Tab 2: Phân Tích & Hồi quy Dữ liệu (NÂNG CẤP TOÀN DIỆN) ---
    with tab2:
        st.header("📊 Phân Tích & Hồi quy Dữ liệu")
        st.write("Upload một file CSV chứa dữ liệu của bạn để bắt đầu phân tích.")

        uploaded_csv = st.file_uploader("Chọn file CSV của bạn", type="csv", key="csv_uploader")

        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.write("**Xem trước dữ liệu gốc:**")
            st.dataframe(df.head())

            # <<< THAY ĐỔI 1: Thực hiện One-Hot Encoding ngay từ đầu
            # Điều này sẽ tạo ra các cột như 'sex_male', 'smoker_yes' để dùng cho cả ma trận tương quan và hồi quy
            df_processed = pd.get_dummies(df, drop_first=True)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # --- 1. BIỂU ĐỒ HISTOGRAM TỪNG BIẾN ---
            st.subheader("1. Phân phối của các biến số (Histogram)")
            selected_col = st.selectbox("Chọn một biến để xem phân phối:", numeric_cols)
            if selected_col:
                # <<< THAY ĐỔI 2: Làm cho biểu đồ nhỏ lại với figsize
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.histplot(df[selected_col], kde=True, ax=ax)
                ax.set_title(f'Phân phối của {selected_col}')
                st.pyplot(fig)

            # --- 2. TƯƠNG QUAN GIỮA CÁC BIẾN ---
            st.subheader("2. Ma trận tương quan giữa các biến")
            st.write(
                "Ma trận này cho thấy mức độ liên quan tuyến tính giữa các biến (bao gồm cả các biến đã được mã hóa).")

            # <<< THAY ĐỔI 3: Tính toán tương quan trên dữ liệu đã được xử lý
            corr = df_processed.corr()
            # <<< THAY ĐỔI 4: Làm cho biểu đồ nhỏ lại với figsize
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, annot_kws={"size": 8})
            st.pyplot(fig_corr)

            # --- 3. HỒI QUY ĐA BIẾN VÀ TÓM TẮT MÔ HÌNH ---
            st.subheader("3. Hồi quy đa biến dự đoán 'charges'")
            if 'charges' not in df_processed.columns:
                st.warning("Vui lòng upload file CSV có chứa cột 'charges' để thực hiện hồi quy.")
            else:
                if st.button("Thực hiện Hồi quy để dự đoán 'charges'"):
                    with st.spinner("Đang xử lý và huấn luyện mô hình..."):
                        # --- Phần code Scikit-learn để dự đoán (giữ nguyên) ---
                        X = df_processed.drop('charges', axis=1)
                        y = df_processed['charges']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
                        sgd_model.fit(X_train_scaled, y_train)
                        y_pred = sgd_model.predict(X_test_scaled)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                        st.success("Huấn luyện mô hình thành công!")

                        # Hiển thị kết quả của Scikit-learn (Mô hình dự đoán)
                        st.write("**Kết quả từ mô hình dự đoán (Scikit-learn - Gradient Descent):**")
                        col1, col2 = st.columns(2)
                        col1.metric("R-squared (R²)", f"{r2:.4f}")
                        col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")

                        # <<< THAY ĐỔI: Gộp và hiển thị các tham số Theta >>>
                        st.write("**Các tham số Theta (θ) tìm được từ Gradient Descent:**")
                        st.write("Đây là các hệ số tối ưu cho mô hình của bạn. Trong đó, Intercept chính là θ₀.")

                        # Lấy các giá trị Theta
                        intercept_theta = sgd_model.intercept_[0]
                        coeffs_theta = sgd_model.coef_

                        # Tạo DataFrame để hiển thị
                        theta_index = ['Intercept (θ₀)'] + list(X.columns)
                        theta_values = [intercept_theta] + list(coeffs_theta)
                        theta_df = pd.DataFrame({'Giá trị Theta (θ)': theta_values}, index=theta_index)

                        st.dataframe(theta_df)

                        # --- Phần code Statsmodels để có Model Summary (MỚI) ---
                        # Thêm cột hệ số chặn cho statsmodels
                        X_sm = sm.add_constant(X)

                        # <<< THAY ĐỔI QUAN TRỌNG: Ép kiểu toàn bộ dữ liệu sang kiểu số >>>
                        X_sm = X_sm.astype(int)

                        # Chạy mô hình OLS
                        model_sm = sm.OLS(y, X_sm).fit()

                        # Hiển thị Model Summary
                        st.write("**Tóm tắt mô hình thống kê (Statsmodels - OLS):**")
                        st.write(
                            "Bảng này cung cấp các chi tiết thống kê về mô hình, như P-value để đánh giá ý nghĩa của từng biến.")
                        # Chuyển bảng tóm tắt thành dạng text để hiển thị trong Streamlit
                        st.text(str(model_sm.summary()))


if __name__ == "__main__":
    main()