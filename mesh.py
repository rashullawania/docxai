from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from typing import Dict, Optional
import os

# Import LangChain components
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image
import re
import pandas as pd
import pdfplumber

# Initialize FastAPI app
app = FastAPI(title="Multi-File RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the MultiFileRAGSystem class
class MultiFileRAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = Ollama(model="llama3.1:8b")

    def process_file(self, file_path):
        """Process a file based on its extension."""
        if file_path.lower().endswith('.pdf'):
            return self.process_pdf(file_path)
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return self.process_image(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
            return self.process_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF, image (JPG/PNG), or Excel file.")

    def process_pdf(self, file_path):
        """Extract and clean text from PDF using pdfplumber."""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([self._clean_text(page.extract_text() or "") for page in pdf.pages])
            texts = self.text_splitter.split_text(text)
            return texts
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    def process_image(self, file_path):
        """Extract and clean text from an image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            text = self._clean_text(text)
            texts = self.text_splitter.split_text(text)
            return texts
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {str(e)}")

    def process_excel(self, file_path):
        """Read and preprocess data from an Excel file."""
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            data_text = df.to_string(index=False)
            data_text = self._clean_text(data_text)
            texts = self.text_splitter.split_text(data_text)
            return texts
        except Exception as e:
            raise RuntimeError(f"Excel processing failed: {str(e)}")

    def _clean_text(self, text):
        """Remove unwanted characters and spaces."""
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        return re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

    def create_vector_store(self, chunks):
        """Create FAISS vector store with embeddings."""
        vector_store = FAISS.from_texts(chunks, self.embeddings)
        return vector_store

    def create_qa_chain(self, vector_store):
        """Create a QA chain with a custom prompt."""
        prompt_template = PromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"]
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template}
        )

# Session Manager for handling user sessions
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.rag_system = MultiFileRAGSystem()

    def create_session(self, file_path: str) -> str:
        session_id = str(uuid.uuid4())
        try:
            chunks = self.rag_system.process_file(file_path)
            vector_store = self.rag_system.create_vector_store(chunks)
            qa_chain = self.rag_system.create_qa_chain(vector_store)
            
            self.sessions[session_id] = {
                "vector_store": vector_store,
                "qa_chain": qa_chain,
                "file_path": file_path
            }
            return session_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_qa_chain(self, session_id: str) -> RetrievalQA:
        session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session["qa_chain"]

# Initialize SessionManager
session_manager = SessionManager()

# Define Pydantic model for question requests
class QuestionRequest(BaseModel):
    question: str
    session_id: str

# API Endpoints
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Create session
        session_id = session_manager.create_session(file_path)
        
        # Cleanup temporary file
        os.remove(file_path)
        
        return {"session_id": session_id, "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        qa_chain = session_manager.get_qa_chain(request.session_id)
        result = qa_chain.invoke({"query": request.question})
        return {"answer": result['result']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/")
async def list_sessions():
    return {"sessions": list(session_manager.sessions.keys())}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    