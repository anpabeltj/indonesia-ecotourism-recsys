import re

STOPWORDS_ID = {
    "yang","untuk","kepada","terhadap","dapat","para","tanpa","bukan","oleh","saat","kami","kamu","mereka",
    "sebagai","adalah","itu","ini","ada","atau","dan","dengan","dari","di","ke","pada","dalam","serta","agar",
    "akan","bila","jika","supaya","karena","tentang","yaitu","yakni","juga","namun","tapi","hanya","saja",
    "lebih","sudah","belum","pernah","sangat","masih","pun","lah","kah","nya","harus","bisa","tentu","mungkin",
    "lalu","kemudian","hingga","sampai","antara","suatu","sebuah","tiap","setiap","banyak","semua","seluruh",
    "berbagai","beberapa","lainnya","lain"
}
IMPORTANT_WORDS = {"di","ke","dari","untuk","dengan","yang"}

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    filtered = [t for t in tokens if (t not in STOPWORDS_ID) or (t in IMPORTANT_WORDS)]
    return " ".join(filtered)
