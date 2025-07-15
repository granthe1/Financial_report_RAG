import os
import streamlit as st
import pandas as pd
import tempfile
import time
from typing import List, Dict, Any
import json
from datetime import datetime
from pathlib import Path
import glob
import re

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

# Config file
CONFIG_FILE = "rag_config.json"

# Default parameters
DEFAULT_CONFIG = {
    "chunk_size": 1500,
    "chunk_overlap": 300,
    "num_chunks": 8,
    "temperature": 0.05,
    "max_tokens": 4000,
    "system_prompt": "You are a financial analyst. Answer strictly based on the provided 10-K documents. If the answer is not in the documents, say you don't know."
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Fill missing keys with default
        for k, v in DEFAULT_CONFIG.items():
            if k not in config:
                config[k] = v
        return config
    else:
        return DEFAULT_CONFIG.copy()

# Load config at startup
CONFIG = load_config()

# Set page config
st.set_page_config(
    page_title="RAG Chatbot - Company Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Claude API Configuration
CLAUDE_API_KEY = 'Claude_API_Key'

# Initialize Claude models
@st.cache_resource
def initialize_models():
    try:
        llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key=CLAUDE_API_KEY,
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"]
        )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

SYSTEM_PROMPT = CONFIG["system_prompt"]

PROMPT_TEMPLATE = """{system_prompt}

Chat History:
{chat_history}

Context Information:
{context}

Question: {question}

Please provide a comprehensive answer based on the context information above. If the information is not available in the context, clearly state that you don't have enough information to answer the question."""

def create_enhanced_prompt():
    return PromptTemplate(
        input_variables=["system_prompt", "chat_history", "context", "question"],
        template=PROMPT_TEMPLATE
    )

@st.cache_data
def load_and_process_pdfs(uploaded_files=None):
    documents = []
    if not uploaded_files:
        documents, pdf_files = load_10k_files_from_folder()
        if not documents:
            return None
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(file.getbuffer())
                try:
                    loader = PyPDFLoader(temp_file_path)
                    file_documents = loader.load()
                    for doc in file_documents:
                        if not doc.metadata:
                            doc.metadata = {}
                        doc.metadata['source_file'] = file.name
                    documents.extend(file_documents)
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")
                    continue
    if not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata['chunk_id'] = i
        chunk.metadata['processed_at'] = datetime.now().isoformat()
    return chunks

@st.cache_resource
def create_vector_store(_chunks, _embeddings):
    if not _chunks or not _embeddings:
        return None
    try:
        vector_store = FAISS.from_documents(_chunks, _embeddings)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": CONFIG["num_chunks"],
                "fetch_k": 25,
                "lambda_mult": 0.6
            }
        )
        return vector_store, retriever
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None

def create_qa_chain(llm, retriever):
    try:
        # Create custom prompt with system prompt
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=f"{CONFIG['system_prompt']}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:"
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": custom_prompt,
                "verbose": False
            }
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

def format_chat_history(messages):
    if not messages:
        return ""
    history = ""
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n\n"
    return history.strip()

def evaluate_response_quality(question, answer, context_docs):
    evaluation = {
        "relevance": 0,
        "completeness": 0,
        "accuracy": 0,
        "sources_used": len(context_docs),
        "response_length": len(answer)
    }
    if "don't have enough information" in answer.lower():
        evaluation["relevance"] = 1
        evaluation["completeness"] = 0
    elif len(answer) > 100:
        evaluation["completeness"] = 1
    if context_docs:
        evaluation["accuracy"] = 1
    return evaluation

def remove_markdown_specials(text):
    # Remove Markdown special characters: *, _, `, $
    return re.sub(r'[*_`$]', '', text)

def main():
    st.title("RAG Chatbot - Company Analysis System")
    st.markdown("### Advanced RAG System using Claude Opus 4 for 10-K Document Analysis")
    st.info("Powered by Claude Opus 4 - The most advanced AI model for superior reasoning and analysis")
    llm, embeddings = initialize_models()
    if not llm or not embeddings:
        st.error("Failed to initialize Claude models. Please check your API key.")
        return
    with st.sidebar:
        st.header("Configuration (from rag_config.json)")
        st.write(CONFIG)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("10-K Document Analysis")
        folder_path = Path("10kFiles")
        has_10k_files = folder_path.exists() and any(folder_path.rglob("*.pdf"))
        if has_10k_files:
            st.success("Found 10-K files in 10kFiles folder. Ready to analyze!")
            pdf_files = list(folder_path.rglob("*.pdf"))
            st.write(f"Available files: {len(pdf_files)}")
            for pdf_file in pdf_files:
                st.write(f"• {pdf_file.name}")
        uploaded_files = st.file_uploader(
            "Upload additional PDF files (optional)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload additional PDF files if needed"
        )
        if "vector_store" not in st.session_state:
            with st.spinner("Processing documents..."):
                chunks = load_and_process_pdfs(uploaded_files)
                if chunks:
                    vector_store, retriever = create_vector_store(chunks, embeddings)
                    if vector_store and retriever:
                        st.session_state.vector_store = vector_store
                        st.session_state.retriever = retriever
                        st.session_state.chunks = chunks
                        st.session_state.retriever.search_kwargs["k"] = CONFIG["num_chunks"]
                        st.success(f"Processed {len(chunks)} chunks from documents!")
                    else:
                        st.error("Failed to create vector store")
                else:
                    st.error("Failed to process documents")
        if "vector_store" in st.session_state:
            st.header("Chat Interface")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            if "evaluation_metrics" not in st.session_state:
                st.session_state.evaluation_metrics = {
                    "total_questions": 0,
                    "avg_relevance": 0,
                    "avg_completeness": 0,
                    "avg_accuracy": 0,
                    "responses": []
                }
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(remove_markdown_specials(message["content"]))
            user_input = None
            if "sample_question" in st.session_state and st.session_state.sample_question:
                user_input = st.session_state.sample_question
                del st.session_state.sample_question
            chat_input = st.chat_input("Ask a question about the companies...")
            if user_input or chat_input:
                if chat_input:
                    user_input = chat_input
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.spinner("Thinking..."):
                    try:
                        qa_chain = create_qa_chain(
                            llm, 
                            st.session_state.retriever
                        )
                        if qa_chain:
                            chat_history = format_chat_history(st.session_state.messages)
                            response = qa_chain.invoke({"query": user_input})
                            answer = response["result"]
                            source_docs = response.get("source_documents", [])
                            evaluation = evaluate_response_quality(user_input, answer, source_docs)
                            st.session_state.evaluation_metrics["total_questions"] += 1
                            st.session_state.evaluation_metrics["responses"].append({
                                "question": user_input,
                                "answer": answer,
                                "evaluation": evaluation,
                                "timestamp": datetime.now().isoformat()
                            })
                            responses = st.session_state.evaluation_metrics["responses"]
                            if responses:
                                avg_relevance = sum(r["evaluation"]["relevance"] for r in responses) / len(responses)
                                avg_completeness = sum(r["evaluation"]["completeness"] for r in responses) / len(responses)
                                avg_accuracy = sum(r["evaluation"]["accuracy"] for r in responses) / len(responses)
                                st.session_state.evaluation_metrics.update({
                                    "avg_relevance": avg_relevance,
                                    "avg_completeness": avg_completeness,
                                    "avg_accuracy": avg_accuracy
                                })
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                                full_response = ""
                                for chunk in answer.split():
                                    full_response += chunk + " "
                                    message_placeholder.write(remove_markdown_specials(full_response))
                                    time.sleep(0.02)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            if source_docs:
                                with st.expander("View Sources"):
                                    for i, doc in enumerate(source_docs):
                                        st.markdown(f"**Source {i+1}**")
                                        st.markdown(f"**Content:** {doc.page_content[:500]}...")
                                        st.markdown(f"**Metadata:** {doc.metadata}")
                                        st.markdown("---")
                            with st.expander("Response Evaluation"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Relevance", evaluation["relevance"])
                                with col2:
                                    st.metric("Completeness", evaluation["completeness"])
                                with col3:
                                    st.metric("Sources Used", evaluation["sources_used"])
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
        else:
            if not has_10k_files:
                st.error("No 10-K files found. Please ensure the 10kFiles folder contains PDF files.")
            else:
                st.info("Processing documents... Please wait.")
    with col2:
        st.header("Quick Actions")
        st.subheader("Sample Questions")
        auto_sample_questions = get_unused_sample_questions()
        if auto_sample_questions:
            sample_questions = auto_sample_questions
        else:
            sample_questions = [
                "Do these companies worry about the challenges or business risks in China or India in terms of cloud service?",
                "How much CASH does Amazon have at the end of 2024?",
                "Compared to 2023, does Amazon's liquidity decrease or increase?",
                "What is the business where main revenue comes from for Amazon, Google, and Microsoft?",
                "What main businesses does Amazon do?",
                "What are the main competitive advantages of Microsoft in cloud computing?",
                "How does Alphabet's advertising revenue compare to other business segments?",
                "What are the key risks mentioned in Amazon's 10-K report?"
            ]
        for i, question in enumerate(sample_questions):
            if st.button(f"Q{i+1}", key=f"sample_{i}"):
                st.session_state.sample_question = question
                st.rerun()
        if st.button("Clear Chat History"):
            if "messages" in st.session_state:
                del st.session_state.messages
            st.rerun()
        if st.button("Export Chat"):
            if "messages" in st.session_state and st.session_state.messages:
                chat_data = {
                    "timestamp": datetime.now().isoformat(),
                    "messages": st.session_state.messages,
                    "evaluation_metrics": st.session_state.evaluation_metrics
                }
                st.download_button(
                    label="Download Chat History",
                    data=json.dumps(chat_data, indent=2),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def load_10k_files_from_folder():
    documents = []
    pdf_files = []
    folder_path = Path("10kFiles")
    if not folder_path.exists():
        st.error("10kFiles folder not found. Please ensure the folder exists with the PDF files.")
        return None, []
    for pdf_file in folder_path.rglob("*.pdf"):
        pdf_files.append(pdf_file)
    if not pdf_files:
        st.error("No PDF files found in 10kFiles folder.")
        return None, []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            file_documents = loader.load()
            for doc in file_documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_path'] = str(pdf_file)
            documents.extend(file_documents)
            st.success(f"Loaded: {pdf_file.name}")
        except Exception as e:
            st.error(f"Error loading {pdf_file.name}: {e}")
            continue
    return documents, pdf_files

def get_unused_sample_questions():
    qa_file = "Q&A.csv"
    eval_files = sorted(glob.glob("rag_eval_results_*.json"), reverse=True)
    if not os.path.exists(qa_file) or not eval_files:
        return None
    import pandas as pd
    try:
        # Try multiple encodings for compatibility
        encodings = ['utf-8-sig', 'gbk', 'latin1']
        for enc in encodings:
            try:
                df = pd.read_csv(qa_file, encoding=enc)
                df = df[df["Question"].notnull() & df["Answer"].notnull()]
                break
            except Exception:
                continue
        else:
            return None
        with open(eval_files[0], "r", encoding="utf-8") as f:
            eval_results = json.load(f)
        used_questions = set(r["question"] for r in eval_results if "question" in r)
        unused = df[~df["Question"].isin(used_questions)]
        if len(unused) == 0:
            return None
        return list(unused["Question"].sample(n=min(8, len(unused)), random_state=42))
    except Exception:
        return None

if __name__ == "__main__":
    main() 