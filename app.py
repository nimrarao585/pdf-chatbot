import os, re, traceback
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ============ PAGE CONFIG ============
st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄")

# ============ API KEY ============
# On Streamlit Community Cloud, set GROQ_API_KEY under "Secrets" in app settings.
# Locally, you can set it as an environment variable before running.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not set. Add it under your app's Secrets (Streamlit Cloud) "
              "or as an environment variable (local).")
    st.stop()
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ============ SESSION STATE ============
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============ PROCESS PDF ============
def process_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever

        return True, f"✅ PDF processed successfully!\n\nPages Loaded: {len(documents)}\nChunks Created: {len(chunks)}"
    except Exception:
        traceback.print_exc()
        return False, f"❌ Error:\n{traceback.format_exc()}"

# ============ CREATE CHAIN ============
def create_chain():
    if st.session_state.retriever is None:
        return False, "❌ Upload and process a PDF first."
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer ONLY from the provided PDF context.
If the answer is not present, reply: "I couldn't find that information in the uploaded PDF."

Context:
{context}

Question:
{input}

Answer:
""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.rag_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
    return True, "✅ Chatbot Ready!"

# ============ ASK QUESTION ============
def ask_question(message):
    rag_chain = st.session_state.rag_chain
    vector_store = st.session_state.vector_store

    if rag_chain is None or vector_store is None:
        return "❌ Please upload and process the PDF first."

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        # Case 1: specific page number, either order ("page 18" or "18 page")
        page_match = re.search(
            r'\bpage\s*(\d+)\b|\b(\d+)\s*(?:st|nd|rd|th)?\s*page\b',
            message.lower()
        )

        if page_match:
            page_num_str = page_match.group(1) or page_match.group(2)
            page_num = int(page_num_str) - 1

            page_chunks = [
                doc.page_content for doc in vector_store.docstore._dict.values()
                if doc.metadata.get("page") == page_num
            ]

            if page_chunks:
                context_text = "\n\n".join(page_chunks)
                prompt = f"Based ONLY on this page content, answer:\n\n{context_text}\n\nQuestion: {message}\n\nAnswer:"
                return llm.invoke(prompt).content
            else:
                return f"❌ Page {page_num + 1} doesn't exist in this PDF."

        # Case 2: normal RAG retrieval
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]

        # Case 3: fallback with bigger k if not found
        if "couldn't find" in answer.lower() or "not present" in answer.lower():
            bigger_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            docs = bigger_retriever.invoke(message)
            if docs:
                context_text = "\n\n".join(d.page_content for d in docs)
                prompt = f"Answer the question using this context:\n\n{context_text}\n\nQuestion: {message}\n\nAnswer:"
                return llm.invoke(prompt).content

        return answer

    except Exception as e:
        traceback.print_exc()
        return f"❌ Error:\n{e}"

# ============ UI ============
st.title("📄 AI PDF Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("📄 Process PDF"):
        with st.spinner("Processing PDF..."):
            # Save uploaded file to a temp path so PyPDFLoader can read it
            temp_path = os.path.join("/tmp", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            ok, msg = process_pdf(temp_path)
            if ok:
                chain_ok, chain_msg = create_chain()
                msg += "\n\n" + chain_msg
            st.session_state.status_message = msg

if "status_message" in st.session_state:
    st.info(st.session_state.status_message)

st.divider()
st.subheader("💬 Chat with your PDF")

# Display chat history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
user_input = st.chat_input("Ask a question about your PDF...")
if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_question(user_input)
            st.markdown(answer)
    st.session_state.messages.append(("assistant", answer))
