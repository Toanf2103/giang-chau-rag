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
class QAData(BaseModel):
    question: str
    answer: str
    category: str
    scope: List[Dict[str, List[str]]]

class QARequest(BaseModel):
    data: List[QAData]
    query: str

# Khởi tạo các biến global
qa_chain = None
embedding = None

def initialize_qa_system(qa_data: List[Dict[str, Any]]):
    global qa_chain, embedding
    
    # Định dạng lại dữ liệu
    formatted_docs = []
    for item in qa_data:
        # Xử lý scope
        position_scope = item['scope'][0]['position'][0]
        department_scope = item['scope'][1]['department'][0]
        
        # Tạo mô tả về phạm vi áp dụng
        scope_desc = f"Áp dụng cho: "
        if position_scope == "*":
            scope_desc += "Tất cả chức vụ"
        else:
            scope_desc += f"Chức vụ: {', '.join(item['scope'][0]['position'])}"
        
        if department_scope == "*":
            scope_desc += ", Tất cả phòng ban"
        else:
            scope_desc += f", Phòng ban: {', '.join(item['scope'][1]['department'])}"
        
        # Tạo văn bản đầy đủ
        doc = f"""
Câu hỏi: {item['question']}
Trả lời: {item['answer']}
{scope_desc}
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
        # Khởi tạo hệ thống với dữ liệu mới
        initialize_qa_system([item.dict() for item in request.data])
        
        # Thực hiện truy vấn
        result = qa_chain(request.query)
        
        return {
            "question": request.query,
            "answer": result['result'],
            "source_documents": [doc.page_content for doc in result['source_documents']]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
