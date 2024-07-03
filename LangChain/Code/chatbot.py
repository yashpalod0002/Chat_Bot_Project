from flask import Flask, render_template, request, jsonify
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__,template_folder="C:\\Users\\yashpalod\\Desktop\\LangChain\\Code\\template")

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(text_chunks):
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    faiss_index = FAISS.from_documents(documents, embeddings)
    print(faiss_index)
    return faiss_index

# Load your PDF and process it once
pdf_path = "C:\\Users\\yashpalod\\Desktop\\LangChain\\Yash Palod Resuma.pdf"  # Path to your PDF file
raw_text = get_pdf_text(pdf_path)
text_chunks = get_text_chunks(raw_text)
faiss_index = get_embeddings(text_chunks)
retriever = faiss_index.as_retriever()
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever
)
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_question = request.json['query']
    
    # Use the chain to answer the user's question
    response = chain.run(question=user_question, chat_history=chat_history)
    
    # Update chat history
    chat_history.append((user_question, response))
    print(response)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
