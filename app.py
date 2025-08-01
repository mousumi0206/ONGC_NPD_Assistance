import streamlit as st

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="📝 ONGC NPD Assistant", layout="wide")

# --- Inject Minecraft-style CSS ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Press Start 2P', cursive;
        background-color: #f0f0f0;
        color: #333;
    }
    h1, h2, h3 {
        color: #2c3e50;
        text-shadow: 1px 1px 0px #fff;
    }
    .stButton>button {
        background-color: #8BC34A;
        color: white;
        border-radius: 0px;
        font-weight: bold;
        border: 2px solid #388E3C;
        box-shadow: 2px 2px 0px #2e7d32;
        transition: transform 0.1s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .stTextInput>div>input {
        border-radius: 0px;
        border: 2px solid #388E3C;
        background-color: #e0f7fa;
        font-family: 'Press Start 2P', cursive;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Imports ---
import os
import shutil
import stat
from dotenv import load_dotenv
from groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# --- Load API key from Streamlit Secrets ---
groq_api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=groq_api_key)

# --- Session State Setup ---
if "doc_chats" not in st.session_state:
    st.session_state.doc_chats = {}
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# --- Sidebar UI ---
st.sidebar.title("🛠️ NPD Assistant Settings")
uploaded_docs = st.sidebar.file_uploader("📄 Upload NPD Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# --- Document Selection Sidebar ---
st.sidebar.markdown("### 📂 Previous Chats")
for doc_name in st.session_state.doc_chats.keys():
    if st.sidebar.button(doc_name):
        st.session_state.selected_doc = doc_name

# --- Embedding Setup ---
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model
Settings.llm = None  # ✅ disables OpenAI fallback

# --- Document Processing ---
doc_dir = "uploaded_docs"
docs = []

if uploaded_docs:
    if os.path.exists(doc_dir):
        shutil.rmtree(doc_dir, onerror=lambda func, path, _: os.chmod(path, stat.S_IWRITE))
    os.makedirs(doc_dir, exist_ok=True)

    for doc in uploaded_docs:
        path = os.path.join(doc_dir, doc.name)
        with open(path, "wb") as f:
            f.write(doc.read())

    current_doc = uploaded_docs[0].name
    st.session_state.selected_doc = current_doc

    if current_doc not in st.session_state.doc_chats:
        st.session_state.doc_chats[current_doc] = []

    reader = SimpleDirectoryReader(doc_dir)
    docs = reader.load_data()

elif os.path.exists("docs") and os.listdir("docs"):
    reader = SimpleDirectoryReader("docs")
    docs = reader.load_data()
else:
    st.warning("📂 No documents uploaded or found in fallback folder.")
    st.stop()

# --- Index and Query Engine ---
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

st.success("✅ Documents processed and indexed!")

# --- Chat Interface ---
st.title("📝 ONGC NPD Assistant")

chat_history = st.session_state.doc_chats.get(st.session_state.selected_doc, [])

user_query = st.text_input("Ask a question about the uploaded documents:")

if user_query:
    chat_history.append({"role": "user", "content": user_query})

    if len(user_query.split()) < 5:
        st.warning("🤔 That seems a bit short. Can you clarify?")
        st.markdown("**Try asking:**")
        st.markdown("- What are the key milestones in the NPD plan?")
        st.markdown("- Who approved the latest revision?")
        st.markdown("- What are the risks mentioned in section 3?")
    else:
        context = query_engine.query(user_query).response
        prompt = f"""You are an NPD assistant. Based on the document context below, answer the user's question clearly and suggest a follow-up question.

Document context:
{context}

User question:
{user_query}
"""
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        ).choices[0].message.content

        chat_history.append({"role": "assistant", "content": response})
        st.session_state.doc_chats[st.session_state.selected_doc] = chat_history

# --- Display Chat History (Newest First) ---
for msg in reversed(chat_history):
    st.chat_message(msg["role"]).write(msg["content"])
