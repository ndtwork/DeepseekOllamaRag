import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from pyvi.ViTokenizer import tokenize ,...

# Define
model_name = "llama3.2:3b-instruct-q8_0"
embedding_model_name = "paraphrase-xlm-r-multilingual-v1"  # Using Sentence Transformers model for multilingual

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
    - **Mô hình nhúng**: Sentence Transformers ({embedding_model_name})
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

    # Tách từ (Tokenize) văn bản tiếng Việt
    tokenized_text = tokenize(full_text)

    # Tạo chunk theo phương pháp overlapping
    chunk_size = 800
    chunk_overlap = 400
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.create_documents([tokenized_text])

    # Cập nhật metadata để tránh lỗi thiếu 'source'
    for doc in documents:
        doc.metadata["source"] = "temp.pdf"

    # Mô hình nhúng bằng Sentence Transformers
    embedder = SentenceTransformer(embedding_model_name)

    # Tạo FAISS vector store
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM
    llm = Ollama(model=model_name)

    # Prompt
    prompt = """
    Bạn là một trợ lý AI chuyên giúp trả lời câu hỏi dựa trên nội dung tài liệu.
    Hãy làm theo các hướng dẫn sau:

    1. Sử dụng thông tin từ tài liệu dưới đây để trả lời câu hỏi.
    2. Nếu không có đủ thông tin, hãy nói "Tôi không biết" thay vì đoán.
    3. Giữ câu trả lời ngắn gọn, súc tích, dễ hiểu.
    4. Trả lời bằng tiếng Việt tự nhiên.

    Ngữ cảnh tài liệu: {context}
    Câu hỏi: {question}
    Câu trả lời chính xác:
    """

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

    # Nhập câu hỏi
    st.header("❓ Đặt câu hỏi về tài liệu")
    user_input = st.text_input("Nhập câu hỏi tại đây:")

    if user_input:
        with st.spinner("Đang xử lý..."):
            try:
                response = qa(user_input)["result"]
                st.success("✅ Trả lời:")
                st.write(response)
            except Exception as e:
                st.error(f"Lỗi xảy ra: {e}")
else:
    st.info("Vui lòng tải lên một tệp PDF để bắt đầu.")
