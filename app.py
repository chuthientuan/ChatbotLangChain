import os
import streamlit as st
import pandas as pd
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# --- Thư viện mới cho Hồi quy Tuyến tính ---
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Cấu hình Vertex AI ---
PROJECT_ID = "vivid-pen-471404-t6"  # Thay thế bằng Project ID của bạn
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


# --- Hàm xử lý cho Hồi quy Tuyến tính (giữ nguyên) ---
def perform_linear_regression(df, x_col, y_col):
    X = df[[x_col]].dropna()
    y = df[[y_col]].dropna()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].values
    y = y.loc[common_index].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    return {
        "model": model, "X": X, "y": y, "y_pred": y_pred,
        "r2": r2, "rmse": rmse, "slope": slope, "intercept": intercept
    }


# --- Giao diện người dùng Streamlit ---
def main():
    st.set_page_config(page_title="🎓 Trợ lý AI Đa năng", page_icon="💡")
    st.title("💡 Trợ lý AI Đa năng")

    tab1, tab2 = st.tabs(["🤖 Chatbot Hỏi Đáp PDF", "📈 Hồi quy Tuyến tính"])

    # --- Giao diện Tab 1: Chatbot ---
    with tab1:
        st.header("Hỏi đáp tài liệu PDF")
        st.write("Upload một file PDF, sau đó đặt câu hỏi về nội dung bên trong.")

        uploaded_pdf = st.file_uploader("Chọn file PDF của bạn", type="pdf", key="pdf_uploader")

        # <<< THAY ĐỔI QUAN TRỌNG 1: Khởi tạo tất cả các biến session_state cần thiết
        if "conversation_chain" not in st.session_state:
            st.session_state.conversation_chain = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "processed_file_name" not in st.session_state:
            st.session_state.processed_file_name = None

        # <<< THAY ĐỔI QUAN TRỌNG 2: Sửa đổi điều kiện để chỉ xử lý file mới
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

                    # <<< THAY ĐỔI QUAN TRỌNG 3: Ghi nhớ tên file đã xử lý
                    st.session_state.processed_file_name = uploaded_pdf.name

                    st.success("Tài liệu đã được xử lý! Bạn có thể bắt đầu hỏi.")

                except Exception as e:
                    st.error(f"Lỗi khi xử lý file: {e}")
                finally:
                    os.remove(uploaded_pdf.name)

        # Hiển thị lịch sử chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Nhận input từ người dùng
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
                    with st.chat_message("assistant"):
                        st.markdown(answer)
            else:
                st.warning("Vui lòng upload một file PDF trước.")

    # --- Giao diện Tab 2: Hồi quy Tuyến tính (giữ nguyên) ---
    with tab2:
        st.header("Tính toán Hồi quy Tuyến tính")
        st.write("Upload một file CSV chứa dữ liệu của bạn để bắt đầu.")
        uploaded_csv = st.file_uploader("Chọn file CSV của bạn", type="csv", key="csv_uploader")

        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.write("**Xem trước dữ liệu:**")
            st.dataframe(df.head())
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_columns) < 2:
                st.warning("File CSV cần ít nhất 2 cột dữ liệu dạng số.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("Chọn biến độc lập (Trục X):", numeric_columns, index=0)
                with col2:
                    y_axis = st.selectbox("Chọn biến phụ thuộc (Trục Y):", numeric_columns, index=1)

                if st.button("Thực hiện Hồi quy"):
                    with st.spinner("Đang tính toán..."):
                        results = perform_linear_regression(df, x_axis, y_axis)
                        st.subheader("Kết quả Hồi quy")
                        st.latex(f"y = {results['slope']:.4f}x + {results['intercept']:.4f}")
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric(label="R-squared (R²)", value=f"{results['r2']:.4f}")
                        with res_col2:
                            st.metric(label="RMSE", value=f"{results['rmse']:.4f}")
                        st.subheader("Biểu đồ Trực quan")
                        fig, ax = plt.subplots()
                        ax.scatter(results['X'], results['y'], alpha=0.7, label="Dữ liệu gốc")
                        ax.plot(results['X'], results['y_pred'], color='red', linewidth=2, label="Đường hồi quy")
                        ax.set_xlabel(x_axis)
                        ax.set_ylabel(y_axis)
                        ax.set_title(f"Hồi quy tuyến tính: {y_axis} vs {x_axis}")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)


if __name__ == "__main__":
    main()