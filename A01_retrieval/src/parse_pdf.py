import os
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def load_cvs(path=DATA_DIR):
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".pdf"):
            candidate_name = filename.split(".")[0]
            pdf_path = os.path.join(path, filename)

            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"

            docs.append({
                "candidate": candidate_name,
                "text": text
            })
    return docs
