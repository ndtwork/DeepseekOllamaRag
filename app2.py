# Thông báo này chỉ ra rằng HuggingFaceEmbeddings trong langchain_community.
# embeddings đã bị deprecate (không còn được khuyến khích dùng) từ phiên bản LangChain 0.2.2,
# và sẽ bị loại bỏ hoàn toàn trong bản 1.0.

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Define
model_name = "llama3.2:3b-instruct-q8_0"
embedding_model_name = "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base"
#embedding_model_name = "keepitreal/vietnamese-sbert"
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
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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


