from sentence_transformers import SentenceTransformer

model_name = "keepitreal/vietnamese-sbert"
model = SentenceTransformer(model_name)

text = "Xin chào, tôi đang thử nghiệm mô hình tiếng Việt."
embedding = model.encode(text)

print(embedding.shape)  # Kiểm tra kích thước vector embedding