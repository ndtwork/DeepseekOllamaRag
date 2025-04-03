import requests
import re
from pdfplumber import open as open_pdf

# Hàm để giải mã URL rút gọn
def expand_url(short_url):
    try:
        response = requests.head(short_url, allow_redirects=True)
        return response.url  # Trả về URL đầy đủ
    except requests.exceptions.RequestException as e:
        return short_url  # Nếu không giải mã được, giữ lại URL gốc

# Đọc nội dung PDF
def process_pdf(file_path):
    with open_pdf(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()

            # Tìm và thay thế các đường link rút gọn trong nội dung
            if text:
                # Tìm các URL trong văn bản (cả URL rút gọn và đầy đủ)
                urls = re.findall(r'https?://[^\s]+', text)
                for url in urls:
                    # Thay thế URL rút gọn bằng URL đầy đủ
                    expanded_url = expand_url(url)
                    text = text.replace(url, expanded_url)

                full_text += text
        return full_text

# Đường dẫn đến tệp PDF
file_path = "D:\\document\\tài liệu link hướng dẫn.pdf"
processed_text = process_pdf(file_path)

# In ra văn bản đã xử lý
print(processed_text)
