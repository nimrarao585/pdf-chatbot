# ============ ALL IMPORTS ============
import os, re, traceback
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ============ GLOBALS ============
vector_store = None
retriever = None
rag_chain = None

# ============ PROCESS PDF ============
def process_pdf(pdf_file):
    global vector_store, retriever
    print("PDF path Gradio gave me:", pdf_file)  # debug line, safe to keep
    if pdf_file is None:
        return "❌ Please upload a PDF."
    try:
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        return f"✅ PDF processed successfully!\n\nPages Loaded: {len(documents)}\nChunks Created: {len(chunks)}"
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error:\n{traceback.format_exc()}"

# ============ CREATE CHAIN ============
def create_chain():
    global rag_chain
    if retriever is None:
        return "❌ Upload and process a PDF first."
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
    rag_chain = create_retrieval_chain(retriever, document_chain)
    return "✅ Chatbot Ready!"

# ============ PROCESS + PREPARE ============
def process_and_prepare(pdf_path):
    if pdf_path is None:
        return "❌ Please upload a PDF."
    msg = process_pdf(pdf_path)
    if "successfully" in msg.lower():
        msg += "\n\n" + create_chain()
    return msg

# ============ ASK QUESTION ============
def ask_question(message, history):
    global rag_chain, vector_store

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

# ============ API KEY ============
# On Render, set GROQ_API_KEY under the "Environment" tab of your service.
# We no longer use getpass() here because a live server has no keyboard input —
# the app would hang forever waiting for you to type a key.
if "GROQ_API_KEY" not in os.environ:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set. "
        "Add it in Render under your service's Environment tab."
    )

# ============ GRADIO APP ============
with gr.Blocks() as demo:
    gr.Markdown("# 📄 AI PDF Chatbot")
    pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
    status = gr.Textbox(label="Status", interactive=False)
    process_btn = gr.Button("📄 Process PDF")
    process_btn.click(fn=process_and_prepare, inputs=pdf, outputs=status)
    gr.ChatInterface(fn=ask_question, title="💬 Chat with your PDF")

# share=True was a Colab-only trick to get a temporary public link.
# On Render, we bind to 0.0.0.0 and the PORT Render assigns us instead.
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
