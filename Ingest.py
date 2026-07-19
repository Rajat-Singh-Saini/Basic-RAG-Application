"""
This script is used for data ingestion. We have uploaded 4 pdf files and stored them into FAISS vector database.
The FAISS vector database is used for similarity search in the RAG application."""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS                 # Vector database for similarity search


load_dotenv() #Load API keys from the .env file

DATA_FOLDER = "documents" # Folder where the 10 HR PDF documents are stored 


VECTOR_DB_PATH = "Course_faiss_index" # Path to save the FAISS vector database for the documents

documents = [] # Empty list to collect all loaded documents

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, file)) # Load the PDF document
        documents.extend(loader.load()) # Add the loaded document to the list

print(f"Loaded {len(documents)} documents.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, # Size of each text chunk in terms of characters
    chunk_overlap=150 # Overlap between chunks to maintain context
)

chunks = text_splitter.split_documents(documents)   

print(f"Split documents into {len(chunks)} chunks.")


embeddings = OpenAIEmbeddings(model = "text-embedding-3-small") # Initialize the OpenAI embeddings model


vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(VECTOR_DB_PATH) # Save the FAISS vector database to disk

print("Course knowledge base created and saved to disk.")