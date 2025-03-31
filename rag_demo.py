import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Khởi tạo FastAPI app
app = FastAPI(title="HR Policy QA System")

# Định nghĩa model cho request
class Policy(BaseModel):
    title: str
    content: str

class QARequest(BaseModel):
    data: List[Policy]
    query: str

# Khởi tạo các biến global
qa_chain = None
embedding = None

def initialize_qa_system(policies: List[Dict[str, str]]):
    global qa_chain, embedding
    
    # Định dạng lại dữ liệu
    formatted_docs = []
    for policy in policies:
        doc = f"""
Chính sách: {policy['title']}
Nội dung: {policy['content']}
"""
        formatted_docs.append(doc)

    # Chunk văn bản
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = text_splitter.create_documents(formatted_docs)

    # Tạo embedding
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Kết nối Qdrant
    qdrant = Qdrant.from_documents(
        documents,
        embedding,
        url="http://localhost:6333",
        prefer_grpc=False,
        collection_name="company-policies"
    )

    # Tạo LLM + Retrieval Chain
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=qdrant.as_retriever(),
        return_source_documents=True
    )

@app.post("/ask")
async def ask_question(request: QARequest):
    try:
        # Chuyển đổi Pydantic model thành dict
        policies_data = [{"title": policy.title, "content": policy.content} for policy in request.data]
        
        # Khởi tạo hệ thống với dữ liệu mới
        initialize_qa_system(policies_data)
        
        # Thực hiện truy vấn
        result = qa_chain(request.query)
        
        return {
            "question": request.query,
            "answer": result['result'],
            "source_documents": [doc.page_content for doc in result['source_documents']]
        }
    except Exception as e:
        print(f"Error details: {str(e)}")  # Thêm log để debug
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
