import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

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

# Khởi tạo embeddings một lần
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

@app.post("/ask")
async def ask_question(request: QARequest):
    try:
        # Chuyển đổi Pydantic model thành định dạng văn bản
        formatted_docs = []
        for policy in request.data:
            doc = f"""
Chính sách: {policy.title}
Nội dung: {policy.content}
"""
            formatted_docs.append(doc)

        # Chunk văn bản
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        documents = text_splitter.create_documents(formatted_docs)

        # Tạo vector store tạm thời trong bộ nhớ sử dụng Chroma
        # Chroma sẽ tự động tạo thư mục tạm thời để lưu trữ
        vectorstore = Chroma.from_documents(
            documents,
            embedding,
            persist_directory=None  # Không lưu trữ bền vững
        )

        # Tạo LLM + Retrieval Chain
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        
        # Thực hiện truy vấn
        result = qa_chain(request.query)
        
        # Trả về kết quả
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
    uvicorn.run(app, host="0.0.0.0", port=3000)