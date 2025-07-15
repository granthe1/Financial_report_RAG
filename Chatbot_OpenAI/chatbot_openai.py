import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


import tempfile
import time 

st.set_page_config(page_title="Chat with Your PDFs (OpenAI)", layout="wide")

st.title("Financial Document Chatbot based on OpenAI")

# API Key Input Section
st.subheader("OpenAI API Key")
api_key = st.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    placeholder="sk-...",
    help="Your API key will be stored securely in this session only."
)

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key to continue.")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []
    
    # Process PDFs if vector store doesn't exist in session state
    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            # Create a temporary directory to save the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    # Save the uploaded file to a temporary file
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Load the PDF
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)

                # Generate embeddings and store in FAISS
                embeddings = OpenAIEmbeddings()
                #embedding_model = SentenceTransformer("FinLang/investopedia_embedding")
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)

        st.success("✅ PDFs uploaded and processed! You can now start chatting.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about your PDFs...")

    prompt_template = """You are a professional financial assistant with expertise in analyzing 10-K filings. Your task is to help users compare Alphabet (Google), Amazon, and Microsoft based on their latest 10-K reports. Use only the information provided in the uploaded 10-K documents to answer user questions accurately.

When answering, follow these guidelines:
– Be fact-based, citing specific sections or data points from the 10-Ks when relevant.
– Avoid speculation or unsupported statements. If information is not available in the 10-Ks, clearly state that.
– Focus on business performance, risk factors, strategy, and financial data as reported in the filings.
– When comparing the companies, highlight similarities and differences with clear structure and supporting evidence.
– Maintain a neutral, professional tone suitable for financial analysts or business users.

Your goal is to be reliable, robust, and insightful — like a high-quality financial analyst assistant. Do not fabricate content. If unsure, state the limitations of the source documents.
If the answer involves a numerical value, such as revenue, expenses, or other metrics, you must extract it only from clearly presented tables or explicitly stated figures in the 10-K documents. Do not calculate or infer values from descriptive narrative text or comparative phrases such as "increased by $X billion." Do not rely on summaries or interpretations. If the information is ambiguous or not explicitly provided, respond by saying: "The information is not explicitly stated in the documents." Do not guess, do not hallucinate, and do not use any external or prior knowledge under any circumstances.

"""

    user_input = prompt_template + user_input if user_input else user_input

    if user_input:
        # Immediately add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o", temperature=0.5),
            retriever=st.session_state.vector_store.as_retriever(),
            chain_type="stuff",
            return_source_documents=True
        )
        
        # Get response from the chatbot with spinner
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": user_input})
            response_text = response["result"]

            if "source_documents" in response:
                with st.expander("Citing from:"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Paragraph {i+1}:**")
                        st.markdown(f"`Page`: {doc.metadata.get('page', 'unknown')}")
                        st.markdown(doc.page_content)
                        st.markdown("---")
        
        # # Display assistant response
        # with st.chat_message("assistant"):
        #     st.markdown(response_text)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate streaming with an existing string
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.05)  # Small delay to simulate streaming
          
        
        # Store assistant response in session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Please upload PDF files to begin.")