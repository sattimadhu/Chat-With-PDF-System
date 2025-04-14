import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Step 1: Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Step 2: Split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Step 3: Create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Save chat logs
def save_log_to_file(question, answer):
    log_data = {
        "timestamp": str(datetime.now()),
        "question": question,
        "answer": answer
    }
    with open("chat_logs.json", "a") as f:
        f.write(json.dumps(log_data) + "\n")

# Step 4: DeepSeek Chat API Call
def deepseek_chat(context, question, api_key):
    prompt = f"""
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, just say "answer is not available in the context".
    Don't make up information.

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
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API Error: {data.get('message', str(e))}"

# Step 5: Handle user question
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    context = "\n".join([doc.page_content for doc in docs])
    response = deepseek_chat(context, user_question, DEEPSEEK_API_KEY)

    save_log_to_file(user_question, response)
    return response

# Main App
def main():
    st.set_page_config("Chat PDF", page_icon="📄", layout="wide")

    # Custom CSS for styling and smaller spacing
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            color: #2c3e50;
            text-align: center;
            font-size: 40px;
            margin-bottom: 10px;
        }
        .subheader {
            color: #16a085;
            font-size: 22px;
        }
        .question, .answer {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 4px;
        }
        .question {
            /*background-color: #ecf0f1;*/
        }
        .answer {
           /* background-color: #d6eaf8;*/
            border-left: 5px solid #2980b9;
        }
        .sidebar-item {
            margin-bottom: 2px;
            padding: 6px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">📚 Chat with PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Upload your PDF files</div>', unsafe_allow_html=True)

    # Upload PDFs in the header area
    pdf_docs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDFs processed and vector store created!")
        else:
            st.warning("Please upload at least one PDF file.")

    # Chat Input
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("💬 Ask a question from the PDF content")

    if user_question:
        with st.spinner("Thinking..."):
            response = user_input(user_question)
            st.session_state.chat_history.append({"question": user_question, "response": response})

    # Display current chat history
    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f'<div class="question"><b>You:</b> {chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer"><b>Bot:</b> {chat["response"]}</div>', unsafe_allow_html=True)

    # Display selected previous chat from sidebar
    if "selected_previous_chat" in st.session_state:
        selected = st.session_state.selected_previous_chat
        st.markdown("<h4>📌 Previously Asked</h4>", unsafe_allow_html=True)
        st.markdown(f'<div class="question"><b>You:</b> {selected["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer"><b>Bot:</b> {selected["answer"]}</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.session_state.pop("selected_previous_chat")

    # Sidebar - Previous Chats
    with st.sidebar:
        st.title("Previous Chats")
        chat_logs = []
        if os.path.exists("chat_logs.json"):
            with open("chat_logs.json", "r") as f:
                for line in reversed(f.readlines()[-20:]):  # Last 20 logs
                    log = json.loads(line)
                    chat_logs.append(log)

        if chat_logs:
            chat_titles = [f"{log['question'][:50]}" for log in chat_logs]
            selected_log = st.selectbox("🔁 Click to view previous Q&A", chat_titles)
            selected_data = chat_logs[chat_titles.index(selected_log)]
            st.session_state.selected_previous_chat = selected_data

if __name__ == "__main__":
    main()
