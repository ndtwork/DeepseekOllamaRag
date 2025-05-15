import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

# === CẤU HÌNH ===
model_name = "llama3.2:3b-instruct-q8_0"
embedding_model_name = "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base"
QDRANT_URL = "https://43965d09-2062-4e87-9d29-fed11a204a3c.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.LjD2RU-VXYXwQ5qPPaAPO6Nhc1mLhozmm8EYaiWi2Cg"
QDRANT_COLLECTION = "quy_che_full"

# === GIAO DIỆN STREAMLIT ===
st.set_page_config(page_title="Chat RAG - Qdrant", layout="wide")
st.title(f"📘 Chatbot RAG với Ollama ({model_name}) & Qdrant")

# Hiển thị phần nhập câu hỏi
user_input = st.text_input("💬 Nhập câu hỏi của bạn:")

# Tải model embedding
embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Kết nối Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Dùng dữ liệu đã có trong Qdrant
vector = Qdrant(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embeddings=embedder,
)

# Tạo retriever
# = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Tạo retriever sử dụng MMR
retriever = vector.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,               # Số đoạn sẽ đưa vào LLM
        "fetch_k": 15,        # Lấy 15 đoạn trước để rerank và chọn ra 5 đoạn đa dạng nhất
        "lambda_mult": 0.5    # 0.0 = thiên về đa dạng, 1.0 = thiên về giống nhau → 0.5 cân bằng
    }
)


# LLM
llm = Ollama(model=model_name)

# Prompt template
prompt = """
1. Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi cuối cùng.
2. Nếu không biết câu trả lời, hãy trả lời: "Tôi không biết" và không bịa.
3. Trả lời chi tiết, rõ ràng, và bằng tiếng Việt.
\n
{context}
\n
Câu hỏi: {question}
Trả lời:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# Xây chuỗi xử lý RAG
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
    return_source_documents=True,
    verbose=True
)

# Nếu người dùng nhập câu hỏi
if user_input:
    with st.spinner("🔍 Đang tìm câu trả lời..."):
        try:
            response = qa(user_input)
            answer = response["result"]
            source_chunks = response["source_documents"]

            st.success("✅ Trả lời:")
            st.write(answer)

            # In ra terminal các chunk được truy xuất
            print("\n===== 🔎 CÁC ĐOẠN ĐƯỢC TRUY XUẤT TỪ QDRANT =====")
            for i, doc in enumerate(source_chunks):
                print(f"\n--- Đoạn {i + 1} ---")
                print(doc.page_content)
                print(f"Source: {doc.metadata.get('source', 'Không rõ')}")

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
else:
    st.info("👉 Nhập câu hỏi vào ô bên trên để bắt đầu.")

# Ghi chú: cải tiếng từ app3 sử dụng kỹ thuật mới MMR