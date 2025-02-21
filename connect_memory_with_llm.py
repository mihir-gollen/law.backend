import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable")

HUGGINGFACE_REPO_ID = "mistral/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    client = InferenceClient(
        model=huggingface_repo_id,
        token=HF_TOKEN
    )
    
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.7,
        model_kwargs={"max_length": 512},  # Remove token here
        huggingfacehub_api_token=HF_TOKEN  # Explicitly pass the API token
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: Let me provide a direct answer based on the context provided.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={
        'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)

# Query handling
def format_source(doc, max_length=200):
    content = doc.page_content
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content

try:
    while True:
        user_query = input("\nWrite Query Here (or 'quit' to exit): ").strip()
        if not user_query:
            continue
        if user_query.lower() == 'quit':
            print("\nGoodbye!")
            break
            
        print("\nProcessing your query...\n")
        try:
            response = qa_chain.invoke({'query': user_query})
            print("ANSWER:")
            print(response["result"])
            print("\nSOURCE DOCUMENTS:")
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"\nSource {i}:")
                print(format_source(doc))
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            continue

except KeyboardInterrupt:
    print("\nOperation cancelled by user")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nPlease ensure you have:")
    print("1. Set your HUGGINGFACE_API_TOKEN environment variable")
    print("2. Have proper access to the Hugging Face model")
    print("3. Have an active internet connection")
    print("\nDetailed error:", str(e))