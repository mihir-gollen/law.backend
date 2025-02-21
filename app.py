import os
import streamlit as st
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

# Streamlit UI Theme
st.set_page_config(page_title="Chatbot Assistant", layout="wide")

# Cache vectorstore loading
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

# Custom prompt for detailed responses
def set_custom_prompt():
    return PromptTemplate(
        template="""
        Use the following context to answer the user's question **in a detailed and structured manner**. 
        If possible, provide examples, explanations, and step-by-step breakdowns.

        Context: {context}

        Question: {question}

        Answer: Please provide a **comprehensive** response, including **details and examples** where applicable.
        """,
        input_variables=["context", "question"]
    )

# Load LLM from Hugging Face
def load_llm():
    HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
    if not HF_TOKEN:
        st.error("Hugging Face API token is missing. Please set it in your environment variables.")
        return None

    HUGGINGFACE_REPO_ID = "mistralai/mistral-7b-instruct-v0.1"

    try:
        llm = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            task="text-generation",
            top_p=0.9,
            temperature=0.6,
            model_kwargs={
                "max_length": 1024,  # Increase response length
                "token": HF_TOKEN
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

# Main Streamlit UI
def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Chatbot Assistant</h1>", unsafe_allow_html=True)
    
    # Create a clear chat button
    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.messages = []

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        role_color = "#E3F2FD" if message['role'] == 'user' else "#F1F8E9"
        st.markdown(f"<div style='background-color:{role_color}; padding:10px; border-radius:10px; margin:5px 0;'>{message['content']}</div>", unsafe_allow_html=True)

    # Get user input
    user_query = st.chat_input("Ask a question...")

    if user_query:
        st.session_state.messages.append({'role': 'user', 'content': f"**You:** {user_query}"})
        st.markdown(f"<div style='background-color:#E3F2FD; padding:10px; border-radius:10px; margin:5px 0;'><b>You:</b> {user_query}</div>", unsafe_allow_html=True)

        # Load vectorstore and LLM
        vectorstore = get_vectorstore()
        llm = load_llm()

        if vectorstore and llm:
            try:
                with st.spinner("Thinking... 🤔"):
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                        chain_type_kwargs={'prompt': set_custom_prompt()}
                    )

                    response = qa_chain.invoke({'query': user_query})

                    # result = response["result"]
                    result = response["result"].encode("utf-8", "ignore").decode("utf-8")
                    result = re.sub(r'[^\x00-\x7F]+', '', result)  # Remove non-ASCII

                    final_response = f"**Assistant:** {result}\n"

                # Display response
                st.session_state.messages.append({'role': 'assistant', 'content': final_response})
                st.markdown(f"<div style='background-color:#F1F8E9; padding:10px; border-radius:10px; margin:5px 0;'><b>Assistant:</b> {result}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
