import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Define
model_name = "llama3.2:3b-instruct-q8_0"
embedding_model_name = "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base"

# App title
st.title(f"📄 Hệ thống RAG với {model_name} & Ollama")

# Sidebar
with st.sidebar:
    st.header("Hướng dẫn")
    st.markdown("""
    1. Tải lên tệp PDF.
    2. Đặt câu hỏi về nội dung trong tài liệu.
    3. Hệ thống sẽ trích xuất và trả lời dựa trên nội dung tài liệu.
    """)
    st.header("Cài đặt")
    st.markdown(f"""
    - **Mô hình nhúng**: HuggingFace ({embedding_model_name})
    - **Kiểu tìm kiếm**: Similarity Search
    - **LLM**: {model_name} (Ollama)
    """)

# Upload file
st.header("📁 Tải lên tài liệu PDF")
uploaded_file = st.file_uploader("Chọn tệp PDF", type="pdf")

if uploaded_file is not None:
    st.success("📄 Tệp PDF đã tải lên thành công! Đang xử lý...")

    # Lưu file tạm thời
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load tài liệu
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    # Tạo chunk theo phương pháp overlapping
    chunk_size = 800
    chunk_overlap = 400
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.create_documents([full_text])

    # Cập nhật metadata để tránh lỗi thiếu 'source'
    for doc in documents:
        doc.metadata["source"] = "temp.pdf"

    # In ra terminal để kiểm tra
    print("\n===== CHUNKS =====")
    for i, doc in enumerate(documents):
        print(f"Chunk {i + 1}:\n{doc.page_content}\n{'-' * 50}")

    # Mô hình nhúng
    embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Tạo FAISS vector store
    vector = FAISS.from_documents(documents, embedder)

    # Lấy 10 đoạn liên quan nhất (tăng k=10)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # LLM để re-rank các đoạn văn bản (có thể sử dụng mô hình LLM để đánh giá và re-rank)
    def re_rank_documents(query, documents, llm):
        # Tạo một danh sách các prompts với các đoạn văn bản và câu hỏi
        re_ranked_docs = []
        for doc in documents:
            prompt = f"""
            Câu hỏi: {query}
            Đoạn văn bản: {doc.page_content}
            Đoạn này có liên quan đến câu hỏi không? (Hãy trả lời ngắn gọn, chỉ rõ mức độ liên quan)
            """
            # Truy vấn LLM với prompt
            response = llm(prompt)
            score = float(response.strip())  # Dự đoán từ LLM, có thể trả về mức độ liên quan dưới dạng điểm số
            re_ranked_docs.append((score, doc))

        # Sắp xếp các đoạn theo mức độ liên quan (theo điểm số)
        re_ranked_docs.sort(reverse=True, key=lambda x: x[0])  # Sắp xếp giảm dần
        return re_ranked_docs[:3]  # Lấy ra 3 đoạn có liên quan nhất

    # LLM
    llm = Ollama(model=model_name)

    # Prompt
    prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
        3. Keep the answer crisp and limited to 3,4 sentences.
        Context: {context}
        Question: {question}
        Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Chuỗi xử lý
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Ngữ cảnh:\n{page_content}\nNguồn: {source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        verbose=True
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        verbose=True,
        return_source_documents=True
    )

    # Nhập câu hỏi từ người dùng
    st.header("❓ Đặt câu hỏi về tài liệu")
    user_input = st.text_input("Nhập câu hỏi tại đây:", key="user_input_key")

    if user_input:
        with st.spinner("Đang xử lý..."):
            try:
                # Lấy 10 đoạn liên quan nhất
                results = qa(user_input)["source_documents"]

                # Re-rank các đoạn này
                re_ranked_docs = re_rank_documents(user_input, results, llm)

                # Hiển thị kết quả
                st.success("✅ Trả lời:")
                for i, (score, doc) in enumerate(re_ranked_docs):
                    st.write(f"**Đoạn {i + 1}** (Độ liên quan: {score}):")
                    st.write(doc.page_content)
                    st.write("-" * 50)
            except Exception as e:
                st.error(f"Lỗi xảy ra: {e}")

else:
    st.info("Vui lòng tải lên một tệp PDF để bắt đầu.")
