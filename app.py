import streamlit as st
from PIL import Image
import io
import requests
import os
from dotenv import load_dotenv

# ---- CONFIG ----
st.set_page_config(
    page_title="Health+Docs AI Hub",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px 20px;
    }
    .stMarkdown {
        color: #2c3e50;
    }
    .feature-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .feature-description {
        font-size: 1.1rem;
        color: #34495e;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .source-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()
API_BASE = os.getenv("RAG_API_BASE", "http://localhost:8000")
CONVO_API_BASE = os.getenv("CONVO_API_BASE", "http://localhost:8001")

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3721/3721127.png", width=100)
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>AI Feature Suite</h1>", unsafe_allow_html=True)
    st.markdown("---")
    feature = st.radio(
        "Choose a Feature",
        ("🩺 Medical Image Analysis", " RAG ChatBot ", "💬 Conversational RAG"),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d;'>
            <small>Powered by Gemini, FAISS, LangChain</small>
        </div>
    """, unsafe_allow_html=True)

# ---- MEDICAL IMAGE ----
def medical_image_analysis():
    st.markdown("<h1 class='feature-title'>🩺 Medical Image Diagnostic Report</h1>", unsafe_allow_html=True)
    st.markdown("<p class='feature-description'>Upload a medical image (X-ray, MRI, CT, etc.) and get an AI-powered diagnostic report instantly.</p>", unsafe_allow_html=True)
    
    upload_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])
    if upload_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(upload_file, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown("### Image Details")
            st.markdown(f"**File Name:** {upload_file.name}")
            st.markdown(f"**File Type:** {upload_file.type}")
            st.markdown(f"**File Size:** {upload_file.size/1024:.2f} KB")
    
    gen_report = st.button("Generate Report", type="primary", use_container_width=True)
    if gen_report and upload_file:
        with st.spinner("Analyzing image and generating report..."):
            import google.generativeai as genai
            GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
            image = Image.open(upload_file).convert("RGB")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()
            try:
                model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
                response = model.generate_content(
                    [
                        "You are a medical imaging specialist. Analyze this medical image and provide a professional diagnostic report including observations, potential findings, and clinical recommendations if any abnormalities are detected.",
                        {"mime_type": "image/png", "data": image_bytes}
                    ],
                    stream=True
                )
                response.resolve()
                report = response.text
                st.success("Analysis Complete!")
                st.markdown("<h2 style='color: #2c3e50;'>📝 Diagnostic Report</h2>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>{report}</div>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="medical_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Try again or check API key configuration.")
    elif gen_report:
        st.warning("Please upload an image first.")

# ---- RAG Q&A ----
def rag_qa():
    st.markdown("<h1 class='feature-title'> RAG ChatBot </h1>", unsafe_allow_html=True)
    st.markdown("<p class='feature-description'>Ask any question related to the documents already uploaded in the knowledge base. Get instant context-aware answers powered by Gemini + FAISS.</p>", unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("📚 Upload Documents", expanded=False):
        uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            if st.button("Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    try:
                        files = [("files", (f.name, f)) for f in uploaded_files]
                        resp = requests.post(f"{API_BASE}/upload_pdf", files=files)
                        if resp.ok:
                            st.success("Documents uploaded successfully!")
                            process_resp = requests.post(f"{API_BASE}/process_pdfs")
                            if process_resp.ok:
                                st.success("Documents processed successfully!")
                            else:
                                st.error("Error processing documents.")
                        else:
                            st.error("Error uploading documents.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    st.markdown("### ❓ Ask a Question")
    query = st.text_input("Type your question related to the knowledge base", placeholder="Enter your question here...")
    if st.button("Get AI Answer", use_container_width=True):
        if not query.strip():
            st.warning("Please type a question!")
        else:
            with st.spinner("Retrieving answer..."):
                try:
                    resp = requests.post(f"{API_BASE}/rag", json={"query": query})
                    if resp.ok:
                        data = resp.json()
                        st.success("Answer:")
                        st.markdown(f"<div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>{data['result']}</div>", unsafe_allow_html=True)
                        with st.expander("📚 View Source Documents"):
                            for idx, doc in enumerate(data["source_documents"], 1):
                                st.markdown(f"<div class='source-box'><b>Source {idx}:</b> {doc['page_content'][:350]}...</div>", unsafe_allow_html=True)
                    else:
                        st.error(resp.json()["detail"] if "detail" in resp.json() else "Failed to get answer.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ---- CONVERSATIONAL RAG ----
def conversational_rag():
    st.markdown("<h1 class='feature-title'>💬 Conversational RAG Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p class='feature-description'>Have a conversation with the AI about medical topics. The AI will remember the context of your conversation.</p>", unsafe_allow_html=True)
    st.markdown("---")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='chat-message user-message'><b>You:</b><br>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message assistant-message'><b>AI:</b><br>{message['content']}</div>", unsafe_allow_html=True)
    query = st.chat_input("Type your message here...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with chat_container:
            st.markdown(f"<div class='chat-message user-message'><b>You:</b><br>{query}</div>", unsafe_allow_html=True)
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{CONVO_API_BASE}/chat", json={"query": query})
                if resp.ok:
                    data = resp.json()
                    response = data["response"]
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    with chat_container:
                        st.markdown(f"<div class='chat-message assistant-message'><b>AI:</b><br>{response}</div>", unsafe_allow_html=True)
                else:
                    st.error(resp.json()["detail"] if "detail" in resp.json() else "Failed to get response.")
            except Exception as e:
                st.error(f"Error: {e}")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        try:
            resp = requests.post(f"{CONVO_API_BASE}/clear_history")
            if resp.ok:
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.error("Failed to clear chat history.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---- MAIN BODY ----
with st.container():
    if feature == "🩺 Medical Image Analysis":
        medical_image_analysis()
    elif feature == " RAG ChatBot ":
        rag_qa()
    else:
        conversational_rag()

# ---- FOOTER ----
st.markdown(
    """
    <hr>
    <center>
    <div style='color: #7f8c8d; font-size: 0.9rem;'>
        Health+Docs AI Suite &copy; 2025 | Powered by Gemini, FAISS, LangChain, Streamlit
        <br>
        <a href="https://github.com/" style='color: #4CAF50; text-decoration: none;'>GitHub</a>
    </div>
    </center>
    """, unsafe_allow_html=True
)
