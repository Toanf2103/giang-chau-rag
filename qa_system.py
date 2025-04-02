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
import google.generativeai as genai

# Load API keys và cấu hình
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_TYPE = os.getenv("MODEL_TYPE", "openai")  # Mặc định là openai

# Cấu hình Gemini nếu cần
if MODEL_TYPE == "gemini":
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Khởi tạo FastAPI app
app = FastAPI(title="HR Policy QA System")

# Định nghĩa model cho request
class Policy(BaseModel):
    id: str
    title: str
    content: str

class QARequest(BaseModel):
    data: List[Policy]
    query: str

# Khởi tạo embeddings cho OpenAI
if MODEL_TYPE == "openai":
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

@app.post("/ask")
async def ask_question(request: QARequest):
    try:
        if MODEL_TYPE == "openai":
            return await handle_openai_request(request)
        elif MODEL_TYPE == "gemini":
            return await handle_gemini_request(request)
        else:
            raise HTTPException(status_code=400, detail="Model type không được hỗ trợ")
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_openai_request(request: QARequest):
    # Chuyển đổi Pydantic model thành định dạng văn bản
    formatted_docs = []
    for policy in request.data:
        doc = f"""
Chính sách: {policy.title}
Nội dung: {policy.content}
ID: {policy.id}
"""
        formatted_docs.append(doc)

    # Chunk văn bản
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = text_splitter.create_documents(formatted_docs)

    # Tạo vector store tạm thời
    vectorstore = Chroma.from_documents(
        documents,
        embedding,
        persist_directory=None
    )

    # Tạo LLM + Retrieval Chain
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    # Thực hiện truy vấn
    result = qa_chain(request.query)
    
    return {
        "question": request.query,
        "answer": result['result'],
        "source_documents": [doc.page_content for doc in result['source_documents']]
    }

async def handle_gemini_request(request: QARequest):
    # Tạo context từ tất cả policies
    context = "\n\n".join([
        f"""
Chính sách: {policy.title}
Nội dung: {policy.content}
ID: {policy.id}
"""
        for policy in request.data
    ])
    
    # Tạo prompt với context
    prompt = f"""
Dựa vào thông tin sau đây:

{context}

Hãy trả lời câu hỏi: {request.query}
"""
    
    # Gọi Gemini API
    response = gemini_model.generate_content(prompt)
    
    return {
        "question": request.query,
        "answer": response.text,
        "source_documents": [context]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 