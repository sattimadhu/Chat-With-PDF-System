import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        page_text = extract_text_directly(doc)
        if not page_text.strip():  # fallback to OCR
            page_text = extract_text_with_ocr(doc)
        text += page_text
    return text

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
# split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# DeepSeek Chat Call with Error Handling
def deepseek_chat(context, question, api_key):
    prompt = f"""
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say "answer is not available in the context". Don't make up information.
    Context: {context}
    Question: {question}

    Answer:
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)

    try:
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return f"API Error: {data['error']['message']}"
        else:
            return f"Unexpected response: {data}"
    except Exception as e:
        return f"Failed to parse response: {e}\nRaw response: {response.text}"

# handle user question and show reply
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    context = "\n".join([doc.page_content for doc in docs])
    response = deepseek_chat(context, user_question, DEEPSEEK_API_KEY)
    #st.write("Reply:", response)
    st.write(response)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# main UI
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF System")
    # local_css("styles.css")
    pdf_docs = st.file_uploader("Upload your PDF files and click Submit & Process", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done!")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)
    
    # with st.sidebar:
    #     st.title("Menu")
        

if __name__ == "__main__":
    main()
