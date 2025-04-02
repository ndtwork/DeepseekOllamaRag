from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

# Chọn mô hình nhúng
#embedding_model_name = "mxbai-embed-large"
#embedding_model_name = "keepitreal/vietnamese-sbert"
embedding_model_name = "VoVanPhuc/sup-SimCSE-Vietnamese-phobert-base"



model = SentenceTransformer(embedding_model_name)

# Danh sách các từ/cụm từ để kiểm tra
phrases = [
    "Tôi yêu lập trình phần mềm và phát triển ứng dụng.",
    "Tôi thích coding, đặc biệt là làm việc với Python.",
    "Học máy là một nhánh của trí tuệ nhân tạo, giúp máy tính tự học từ dữ liệu.",
    "Trí tuệ nhân tạo đang thay đổi thế giới công nghệ.",
    "Bóng đá là môn thể thao vua được yêu thích trên toàn cầu.",
    "Môn thể thao vua có hàng triệu người hâm mộ khắp thế giới."
]
# Chuyển thành vector
vectors = model.encode(phrases)
print(f"Kích thước vector: {vectors.shape}")  # (số câu, số chiều embedding)

# Tính độ tương tự Cosine giữa các cụm từ
def similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

print("Độ tương tự giữa các cụm từ:")
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        sim = similarity(vectors[i], vectors[j])
        print(f"({phrases[i]}) <-> ({phrases[j]}) = {sim:.4f}")


for i, phrase in enumerate(phrases):
    print(f"{phrase}: {vectors[i][:5]}")


# Vẽ biểu đồ
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Giảm số chiều vector từ 768 -> 2 để vẽ
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Vẽ scatter plot
plt.figure(figsize=(8,6))
for i, txt in enumerate(phrases):
    plt.scatter(reduced_vectors[i,0], reduced_vectors[i,1])
    plt.annotate(txt, (reduced_vectors[i,0], reduced_vectors[i,1]))

plt.title("Biểu diễn vector sau khi giảm chiều (PCA)")
plt.show()

