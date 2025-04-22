import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

st.set_page_config(page_title="Multi PDF Text Extractor", layout="wide")
st.title("📄 Smart Multi-PDF Text Extractor (OCR + Selectable)")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

def extract_text_directly(doc):
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_with_ocr(doc):
    text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes()))
        text += pytesseract.image_to_string(img) + "\n\n"
    return text.strip()

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### 📂 {uploaded_file.name}")
        st.info("Analyzing PDF...")
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = extract_text_directly(doc)
        if not text:
            text = extract_text_with_ocr(doc)

        if text:
            st.text_area(f"📝 Extracted Text from {uploaded_file.name}", text, height=400)
        else:
            st.error("❌ No text could be extracted from this PDF.")
