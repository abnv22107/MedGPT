import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.memory import ConversationSummaryMemory
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set HuggingFace Hub API key
# os.environ["HF_TOKEN"] = "your-hf-token-here"  # Replace with your token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"  # or "tiiuae/falcon-7b-instruct"



# Initialize FastAPI app
app = FastAPI(title="Conversational RAG API")

try:
    # Load FAISS vectorstore
    logger.info("Loading FAISS vectorstore...")
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    logger.info("FAISS vectorstore loaded successfully")

    # Load Mistral-7B-Instruct from Hugging Face Hub
    logger.info("Initializing HuggingFace model...")
    # llm = HuggingFaceEndpoint(
    #     repo_id=HUGGINGFACE_REPO_ID,
    #     task="conversational",
    #     temperature=0.5,
    #     max_new_tokens=512,
    #     top_p=0.95,
    #     repetition_penalty=1.15,
    #     huggingfacehub_api_token=HF_TOKEN
    # )
    llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512,
    top_p=0.95,
    repetition_penalty=1.15,
    huggingfacehub_api_token=HF_TOKEN
)
    logger.info("HuggingFace model initialized successfully")

    # Memory with summarization
    memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True,
        memory_key="chat_history"
    )

    # Custom system prompt
    system_prompt = """
    You are Med-bot, an AI medical assistant. Use the chat history and the retrieved medical information
    below to answer user questions in a helpful, medically-reasonable way. Be cautious and recommend
    consulting a real doctor for any serious condition.
    Mind that you are just a medical expert and you don't have any knowledge about any other content the user is asking, If asked so just say you can't provide any information on that topic and content.
    In the follow up questions too don't provide any other content except the medical content.
    """

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt + "\n\nRelevant Information:\n{context}\n\nUser Question: {question}\nAnswer:"
    )

    # Build Conversational RAG chain
    logger.info("Building RAG chain...")
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True
    )
    logger.info("RAG chain built successfully")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

# Pydantic model for request
class ChatRequest(BaseModel):
    query: str

# Endpoint for chat
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.query}")
        response = rag_chain.invoke({"question": request.query})
        logger.info("Successfully generated response")
        return {
            "response": response["answer"],
            "chat_history": memory.chat_memory.messages
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to clear chat history
@app.post("/clear_history")
async def clear_history():
    try:
        logger.info("Clearing chat history")
        memory.clear()
        return {"status": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")




        