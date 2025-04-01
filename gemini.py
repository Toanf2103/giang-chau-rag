import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Khởi tạo FastAPI app
app = FastAPI(title="HR Policy QA System with Gemini")

# Định nghĩa model cho request
class Policy(BaseModel):
    title: str
    content: str

class QARequest(BaseModel):
    data: List[Policy]
    query: str

# Khởi tạo biến global
model = None

def initialize_qa_system(policies: List[dict]):
    global model
    # Khởi tạo Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')

@app.post("/ask")
async def ask_question(request: QARequest):
    try:
        # Chuyển đổi Pydantic model thành dict
        policies_data = [{"title": policy.title, "content": policy.content} 
                        for policy in request.data]
        
        # Khởi tạo hệ thống với model
        initialize_qa_system(policies_data)
        
        # Tạo context từ tất cả policies
        context = "\n\n".join([
            f"Chính sách: {policy['title']}\nNội dung: {policy['content']}"
            for policy in policies_data
        ])
        
        # Tạo prompt với context
        prompt = f"""
Dựa vào thông tin sau đây:

{context}

Hãy trả lời câu hỏi: {request.query}
"""
        
        # Gọi Gemini API
        response = model.generate_content(prompt)
        
        return {
            "question": request.query,
            "answer": response.text,
            "source_documents": [context]
        }
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)