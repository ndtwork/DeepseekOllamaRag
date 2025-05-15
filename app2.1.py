import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Define models
model_name = "llama3.2:3b-instruct-q8_0"
embedding_model_name = "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base"

# App title
st.title(f"\U0001F4C4 Hệ thống RAG với {model_name} & Ollama")

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
st.header("\U0001F4C1 Tải lên tài liệu PDF")
uploaded_file = st.file_uploader("Chọn tệp PDF", type="pdf")

if uploaded_file is not None:
    st.success("\U0001F4C4 Tệp PDF đã tải lên thành công! Đang xử lý...")

    # Lưu file tạm thời
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load tài liệu
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    # Tạo chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400)
    documents = text_splitter.create_documents([full_text])

    # Gắn metadata
    for doc in documents:
        doc.metadata["source"] = "temp.pdf"

    # Tạo vector store
    embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 15})

    # LLM và Prompt
    llm = Ollama(model=model_name)
    prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
        3. Trả lời chi tiết và đầy đủ dựa trên ngữ cảnh. Không giới hạn độ dài.
        Context: {context}
        Question: {question}
        Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Kết hợp documents chain
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Ngữ cảnh:\n{page_content}\nNguồn: {source}"
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
                response = qa(user_input)
                answer = response["result"]
                source_chunks = response["source_documents"]

                # Re-ranking bằng LLM
                ranked_chunks = []
                for chunk in source_chunks:
                    chunk_content = chunk.page_content
                    ranking_prompt = f"""
                        Trả về một số điểm (từ 0.0 đến 1.0) thể hiện mức độ liên quan giữa đoạn sau và câu hỏi:

                        Câu hỏi: {user_input}
                        Đoạn văn bản: {chunk_content}

                        Chỉ trả về một số duy nhất."""
                    ranking_response = llm.invoke(ranking_prompt).strip()

                    try:
                        score = float(ranking_response)
                    except:
                        score = 0.0

                    ranked_chunks.append((chunk_content, score))

                ranked_chunks.sort(key=lambda x: x[1], reverse=True)
                top_3_chunks = ranked_chunks[:3]

                st.success("✅ Trả lời:")
                st.write(answer)

                st.markdown("### 🔍 Top 3 đoạn văn bản liên quan:")
                for i, (chunk, score) in enumerate(top_3_chunks):
                    st.markdown(f"**Đoạn {i + 1} (score={score:.2f})**:")
                    st.markdown(f"> {chunk}")

            except Exception as e:
                st.error(f"Lỗi xảy ra: {e}")
else:
    st.info("Vui lòng tải lên một tệp PDF để bắt đầu.")


# sử dụng klyx th