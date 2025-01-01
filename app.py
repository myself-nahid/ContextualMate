import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import csv
from dotenv import load_dotenv

load_dotenv()

# Set environment variables
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Embeddings and LLM initialization
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

# Chat Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Preprocess CSV to handle errors
def preprocess_csv(file_path):
    try:
        # Read the file and skip bad lines
        df = pd.read_csv(file_path, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        st.write(f"Loaded {df.shape[0]} rows successfully")
        
        # Save cleaned file
        cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")
        df.to_csv(cleaned_file_path, index=False)
        st.write(f"Cleaned CSV saved at: {cleaned_file_path}")
        return cleaned_file_path
    except Exception as e:
        st.error(f"Error preprocessing CSV: {e}")
        st.stop()

# Create Vector Embeddings
def create_vector_embedding():
    original_file_path = r"D:\TRIL-INTERN\TASK01\preprocess-scraped-data\cleaned_genzmarketing_data.csv"
    cleaned_file_path = preprocess_csv(original_file_path)
    
    if "vectors" not in st.session_state:
        try:
            st.session_state.loader = CSVLoader(file_path=cleaned_file_path)
            st.session_state.docs = st.session_state.loader.load()
            st.write(f"Loaded {len(st.session_state.docs)} documents successfully")
            
            # Split documents and create vector embeddings
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embeddings)
            st.write("Vector database created successfully!")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            st.stop()

# Streamlit UI
st.title("GenZ Contextual Chatbot: AI-Powered Solution for Accurate and Insightful Query Responses")

# User query input
user_prompt = st.text_input("Enter your query from the GenZMarketing website.")

# Initialize vector embedding
if st.button("Document Embedding"):
    create_vector_embedding()

# Check if vectors exist before processing query
if user_prompt and "vectors" in st.session_state:
    try:
        # Create document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Retrieve response
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(response['answer'])

        # Show related documents
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
    except Exception as e:
        st.error(f"Error processing query: {e}")
else:
    st.warning("Vector database not initialized. Please run 'Document Embedding' first.")
