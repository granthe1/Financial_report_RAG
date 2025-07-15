# Technical Note

## System Architecture

This project is a modular suite of AI-powered PDF document chatbots, each leveraging a different state-of-the-art large language model (LLM): OpenAI GPT-4o, Google Gemini-2.0-flash, and Anthropic Claude-3. Each chatbot is implemented as a Streamlit web application, providing a user-friendly interface for document upload, processing, and interactive Q&A. The architecture separates document ingestion, chunking, embedding, vector storage, retrieval, and LLM inference, ensuring flexibility and extensibility.

## Model Selection Rationale

- **OpenAI GPT-4o (Chatbot_OpenAI):** Chosen for its strong performance in general and financial language understanding, and seamless integration with LangChain and OpenAI APIs.
- **Google Gemini-2.0-flash (Gemini):** Selected to provide diversity in LLM behavior and to leverage Google's generative AI capabilities, especially for document-grounded Q&A.
- **Anthropic Claude-3 (Claude_Model):** Integrated for its advanced reasoning and safety features, and to enable Retrieval-Augmented Generation (RAG) workflows with flexible configuration.

This multi-model approach allows users to compare LLM performance and select the best tool for their specific document analysis needs.

## Data Flow and Processing

1. **PDF Upload and Parsing:** Users upload one or more PDF files via the Streamlit interface. The system uses `PyPDFLoader` to extract text from each document.
2. **Text Chunking:** Extracted text is split into overlapping chunks (chunk size and overlap configurable per model) to preserve context and improve retrieval accuracy.
3. **Embedding and Vector Storage:** Each chunk is embedded using the model-specific embedding method (OpenAI, Google, or HuggingFace) and stored in a FAISS vector database for efficient similarity search.
4. **Retrieval and Q&A:** Upon user query, the system retrieves the most relevant document chunks using vector similarity. The selected LLM generates an answer strictly based on the retrieved content, following a carefully engineered prompt to avoid hallucination.
5. **Response and Citation:** The answer is displayed to the user, with citations or excerpts from the source documents for transparency and traceability. Multi-turn chat history is supported in all models.

## Key Implementation Details

- **Prompt Engineering:** Each chatbot uses a persona-driven prompt to ensure answers are grounded in the uploaded documents. Prompts are tailored for financial analysis or general helpfulness, and instruct the LLM to admit when information is insufficient.
- **Session and State Management:** Streamlit's session state is used to manage chat history, vector stores, and user API keys securely within each session.
- **Configuration:** The Claude_Model supports external configuration via `rag_config.json`, allowing users to adjust chunking, retrieval, and prompt parameters.
- **Error Handling:** Robust error handling is implemented for file uploads, document parsing, and API interactions, with informative feedback to users.
- **Security:** API keys are never hardcoded (except as placeholders) and are either input by the user or set via environment variables. Users are advised not to upload sensitive documents.

## Conclusion

This project demonstrates a practical, extensible framework for document-grounded Q&A using multiple leading LLMs. By combining advanced retrieval techniques, configurable chunking, and careful prompt engineering, it enables users to extract reliable insights from complex PDF documents in a transparent and user-friendly manner. 