import os
import streamlit as st
import pdfplumber
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import torch
import re

torch.set_default_dtype(torch.float32)
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(page_title="Legal AI Assistant", layout="wide")

st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 32px;
            color: #2c3e50;
            font-weight: bold;
        }
        .chat-container {
            padding: 10px;
        }
        .user-message {
            background-color: #ecf0f1;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .assistant-message {
            background-color: #d5e8d4;
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={"normalize_embeddings": True}
    )
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt():
    return PromptTemplate(
        template="""
        Use the following context to answer the user's question **in a professional and structured manner**.
        Provide legal explanations with case references and clear formatting.

        Context: {context}

        Question: {question}

        Answer: Please ensure clarity, accuracy, and use structured formatting.
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        st.error("Hugging Face API token is missing. Please set it in your environment variables.")
        return None
    
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    
    try:
        llm = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            task="text-generation",
            top_p=0.9,
            temperature=0.6,
            model_kwargs={"max_length": 1024},
            huggingfacehub_api_token=HF_TOKEN
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

def extract_text_from_pdf(uploaded_file):
    extracted_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text.strip()

def detect_risky_clauses(text):
    risky_keywords = [
        "waiver of rights", "non-compete", "liquidated damages", "unilateral termination", "indemnification",
        "binding arbitration", "confession of judgment", "penalty clause", "severability", "excessive liability"
    ]
    
    detected_clauses = [clause for clause in risky_keywords if clause in text.lower()]
    
    if detected_clauses:
        return f"⚠️ The contract contains potentially risky clauses: {', '.join(detected_clauses)}. Consider reviewing these sections carefully."
    return "✅ No highly risky clauses detected in the document."

def main():
    st.markdown("<h1 class='title'>⚖️ Legal AI Assistant</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("📂 Upload Legal Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    document_text = ""
    if uploaded_file:
        document_text = extract_text_from_pdf(uploaded_file)
        st.sidebar.success("✅ Document Uploaded Successfully!")
        st.sidebar.text_area("Extracted Text", document_text, height=150)
        st.sidebar.markdown(detect_risky_clauses(document_text))
    
    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.messages = []
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        role_class = "user-message" if message['role'] == 'user' else "assistant-message"
        st.markdown(f"<div class='{role_class}'><b>{message['role'].capitalize()}:</b> {message['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    user_query = st.chat_input("Ask a legal question...")
    
    if user_query:
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        st.markdown(f"<div class='user-message'><b>You:</b> {user_query}</div>", unsafe_allow_html=True)

        vectorstore = get_vectorstore()
        llm = load_llm()

        if vectorstore and llm:
            try:
                with st.spinner("Processing your query... ⚖️"):
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        chain_type_kwargs={'prompt': set_custom_prompt()}
                    )
                    combined_query = f"{document_text[:2000]}\n\nUser Query: {user_query}" if document_text else user_query
                    response = qa_chain.invoke({'query': combined_query})
                    result = re.sub(r'<.*?>', '', response["result"]).encode("utf-8", "ignore").decode("utf-8")
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                st.markdown(f"<div class='assistant-message'><b>Assistant:</b> {result}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
