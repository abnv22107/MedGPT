import os
import logging
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.responses import JSONResponse
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load environment variables (.env)
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# 2. FastAPI instance
app = FastAPI()

# 3. Load Vector Database and Embeddings (at startup)
try:
    logger.info("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    logger.info("Loading vector store...")
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully")
    else:
        logger.warning("Vector store not found. Please upload and process PDFs first.")
        db = None

    logger.info("Initializing Gemini model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5,
        max_output_tokens=512,
        convert_system_message_to_human=True
    )

    # 4. Custom Prompt
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you dont know, dont try to make up an answer.
    Dont provide anything out of the given context
    And let the answer be a bit more detailed and specific not too small.

    Context: {context}
    Input: {input}

    Start the answer directly. No small talk please.
    """
    custom_prompt = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)

    # 5. Create chains
    if db:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        document_chain = create_stuff_documents_chain(llm, custom_prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)
        logger.info("QA chain created successfully")
    else:
        qa_chain = None
        logger.warning("QA chain not created - no vector store available")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# 6. Pydantic request model
class QueryRequest(BaseModel):
    query: str

@app.post("/upload_pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    try:
        os.makedirs(DATA_PATH, exist_ok=True)
        saved_files = []
        for file in files:
            file_location = os.path.join(DATA_PATH, file.filename)
            with open(file_location, "wb") as f:
                f.write(await file.read())
            saved_files.append(file.filename)
        return {"uploaded": saved_files}
    except Exception as e:
        logger.error(f"Error uploading PDFs: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_pdfs")
def list_pdfs():
    try:
        if not os.path.exists(DATA_PATH):
            return {"pdfs": []}
        pdfs = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
        return {"pdfs": pdfs}
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_pdfs")
def process_pdfs():
    try:
        # Step 1: Load PDFs
        loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            return JSONResponse(status_code=400, content={"error": "No PDFs found to process."})

        # Step 2: Chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)

        # Step 3: Embedding
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Step 4: Store in FAISS
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        return {"status": "success", "chunks": len(text_chunks)}
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    try:
        exists = os.path.exists(DB_FAISS_PATH)
        return {"vectorstore_exists": exists}
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# 7. RAG endpoint
@app.post("/rag")
def rag_query(req: QueryRequest):
    try:
        logger.info(f"Received query: {req.query}")
        
        if not db or not qa_chain:
            logger.error("Vector store or QA chain not initialized")
            raise HTTPException(
                status_code=400,
                detail="Vector store not initialized. Please upload and process PDFs first."
            )
        
        # Run QA chain
        logger.info("Running QA chain...")
        result = qa_chain.invoke({"input": req.query})
        logger.info("QA chain completed successfully")
        
        return {
            "result": result['answer'],
            "source_documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result['context']
            ]
        }
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
