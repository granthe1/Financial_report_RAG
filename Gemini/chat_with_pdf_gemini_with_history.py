import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import time
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# -----------------------------------
# 1. Persona (Balanced)
# -----------------------------------
persona = """
You are a careful and helpful assistant. Your job is to answer questions using the provided documents.
Prioritize information from the documents, and avoid introducing facts not supported there.
If the documents do not contain enough detail to answer confidently, it's okay to say you don't have enough information.
You can summarize or rephrase document text in clear, helpful language.
Cite specific parts or page numbers if possible.
"""

# -----------------------------------
# 2. Prompt Template (Balanced)
# -----------------------------------
template = """
{persona}

-----------------------
Chat History:
{chat_history}

-----------------------
DOCUMENT EXCERPTS:
{context}

-----------------------
Based only on the DOCUMENT EXCERPTS above and considering the Chat History,
answer the following question:

Question: {question}

If the DOCUMENT EXCERPTS do not contain enough information, say:
"I don't have enough information to answer this question."
"""

# -----------------------------------
# 3. API Key Setup
# -----------------------------------
GOOGLE_API_KEY = '————————————'  # Replace with your actual Gemini API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# -----------------------------------
# 4. LLM Setup (lower temperature for less hallucination)
# -----------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3
)

# -----------------------------------
# 5. Streamlit Page Setup
# -----------------------------------
st.set_page_config(page_title="Chat with Your PDFs (Gemini) Version 2")

# Show central success message on main screen if flagged
if st.session_state.get("reset_success", False):
    st.success("✅ You are good to go!")
    st.session_state["reset_success"] = False

# Sidebar with Session Controls
with st.sidebar:
    st.header("Session Controls")
    st.markdown("Manage your session below:")

    # Apply uniform button styling with CSS
    st.markdown("""
        <style>
        div.stSidebar button {
            width: 100% !important;
            padding: 0.75em 0.5em;
            font-weight: 600;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("Shut Down"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["reset_success"] = True
        st.rerun()

    if st.button("Shut Down + Clear Cache"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["reset_success"] = True
        # Clear Streamlit cache
        st.cache_data.clear()
        st.rerun()

# Main Header Container
with st.container():
    st.markdown("""
    # Chat with Your PDFs (Gemini) Version 2
    
    Upload your PDF reports and ask questions about them.  
    Get clear answers grounded in your documents.
    """)


# -----------------------------------
# 6. PDF Upload
# -----------------------------------
with st.container():
    st.subheader("Upload your PDF files")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here (200MB max each):",
        accept_multiple_files=True,
        type=["pdf"]
    )

if uploaded_files:
    documents = []

    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file in uploaded_files:
                        temp_file_path = os.path.join(temp_dir, file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(file.getbuffer())
                        loader = PyPDFLoader(temp_file_path)
                        documents.extend(loader.load())

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, 
                        chunk_overlap=50,
                        length_function=len
                    )
                    docs = text_splitter.split_documents(documents)

                    if docs:
                        st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                        st.session_state.processed_files = [file.name for file in uploaded_files]
                    else:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        st.stop()
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")
                st.stop()

        with st.container():
            st.success("PDFs uploaded and processed! You can now start chatting.")

    # -----------------------------------
    # 7. Chat History Display
    # -----------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    with st.container():
        st.subheader("Chat History")
        for message in st.session_state.messages:
            role_label = "**User:**" if message["role"] == "user" else "**Assistant:**"
            st.markdown(f"{role_label} {message['content']}")

    # -----------------------------------
    # 8. User Input Section
    # -----------------------------------
    with st.container():
        user_input = st.chat_input("Ask a question about your PDFs...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.container():
            st.markdown(f"**User:** {user_input}")

        # -----------------------------------
        # 9. Retriever (moderate k, balanced)
        # -----------------------------------
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.7}
        )

        # -----------------------------------
        # 10. Create Conversational Chain
        # -----------------------------------
        try:
            # Create the conversational retrieval chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                return_source_documents=True,
                verbose=False
            )

            # -----------------------------------
            # 11. Get Response
            # -----------------------------------
            with st.spinner("Thinking..."):
                response = qa_chain({"question": user_input})
                response_text = response["answer"]

                # Display source documents
                if "source_documents" in response and response["source_documents"]:
                    with st.expander("Retrieved Context from Documents"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.markdown(doc.page_content)
                            page_num = doc.metadata.get('page', 'unknown')
                            source_file = doc.metadata.get('source', 'unknown')
                            st.caption(f"Page: {page_num} | Source: {os.path.basename(source_file) if source_file != 'unknown' else 'unknown'}")

            # -----------------------------------
            # 12. Display Assistant Answer with typing effect (Part 14 from original)
            # -----------------------------------
            with st.container():
                message_placeholder = st.empty()
                full_response = ""
                
                # Typing effect - exactly like your original part 14
                for chunk in response_text.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(f"**Assistant:** {full_response}")
                    time.sleep(0.05)

            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
            st.error("Please try rephrasing your question or check if your PDFs were processed correctly.")

else:
    with st.container():
        st.info("Please upload PDF files to begin.")

# -----------------------------------
# 13. Display processed files info
# -----------------------------------
if "processed_files" in st.session_state:
    with st.sidebar:
        st.subheader("Processed Files")
        for file_name in st.session_state.processed_files:
            st.write(f"📄 {file_name}")