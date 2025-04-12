import numpy as np
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize

# Khởi tạo model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Danh sách câu
sentences = [
    'Kẻ đánh bom đinh tồi tệ nhất nước Anh.',
    'Nghệ sĩ làm thiện nguyện - minh bạch là việc cấp thiết.',
    'Bắc Giang tăng khả năng điều trị và xét nghiệm.',
    'HLV futsal Việt Nam tiết lộ lý do hạ Lebanon.',
    'việc quan trọng khi kêu gọi quyên góp từ thiện là phải minh bạch, giải ngân kịp thời.',
    '20% bệnh nhân Covid-19 có thể nhanh chóng trở nặng.',
    'Thái Lan thua giao hữu trước vòng loại World Cup.',
    'Cựu tuyển thủ Nguyễn Bảo Quân: May mắn ủng hộ futsal Việt Nam',
    'Chủ ki-ốt bị đâm chết trong chợ đầu mối lớn nhất Thanh Hoá.',
    '1/5 bệnh nhân covid có thể gặp tình trạng tồi tệ'
]

# Tiền xử lý (tokenization)
sentences = [tokenize(sentence) for sentence in sentences]

# Encode thành vector embeddings
embeddings = model.encode(sentences)

# Tính ma trận khoảng cách Euclid giữa các vector
euclidean_distances = cdist(embeddings, embeddings, metric='euclidean')

# Trả về ma trận khoảng cách Euclid
euclidean_distances


import pandas as pd

# Chuyển ma trận khoảng cách thành DataFrame để hiển thị đẹp hơn
df_euclidean = pd.DataFrame(
    euclidean_distances,
    columns=[f"Câu {i+1}" for i in range(len(sentences))],
    index=[f"Câu {i+1}" for i in range(len(sentences))]
)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"Khoảng cách giữa Câu {i+1} và Câu {j+1}: {euclidean_distances[i][j]:.4f}")
