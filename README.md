# AI-Powered PDF Chatbot Suite

## Project Overview
This repository provides a suite of AI-powered PDF document chatbots, enabling users to upload PDF files (such as financial reports) and interact with their content using natural language. The system supports three leading LLM providers:
- **OpenAI GPT-4o** (Chatbot_OpenAI)
- **Google Gemini-2.0-flash** (Gemini)
- **Anthropic Claude-3** (Claude_Model)

Each model is wrapped in a user-friendly Streamlit web interface, supporting document upload, chunking, vector search, and context-grounded Q&A.

## Features
- Upload and parse multiple PDF files
- Document chunking and vector storage (FAISS)
- Context-aware intelligent Q&A
- Multi-turn conversation with chat history
- Source citation for answers
- Streamlit-based user interface
- Secure API key input (where required)

## Model Details

### Chatbot_OpenAI (OpenAI GPT-4o)
- **LLM**: GPT-4o via `langchain_openai.ChatOpenAI`
- **Embedding Model**: OpenAIEmbeddings
- **Vector Store**: FAISS
- **Prompt**: Financial analyst persona, strict document grounding
- **API Key**: User inputs OpenAI API key at runtime

### Gemini (Google Gemini-2.0-flash)
- **LLM**: Gemini-2.0-flash via `langchain_google_genai.ChatGoogleGenerativeAI`
- **Embedding Model**: GoogleGenerativeAIEmbeddings
- **Vector Store**: FAISS
- **Prompt**: Helpful assistant persona, document-grounded, admits when information is insufficient
- **API Key**: Set in code or via environment variable

### Claude_Model (Anthropic Claude-3)
- **LLM**: Claude-3 via `langchain_anthropic.ChatAnthropic`
- **Embedding Model**: HuggingFaceEmbeddings (MiniLM)
- **Vector Store**: FAISS
- **Prompt**: Financial analyst persona, strict document grounding
- **Configurable**: Loads parameters from `rag_config.json`
- **API Key**: Set in code or via environment variable

## Requirements
- Python 3.8+
- Main dependencies:
  - streamlit
  - langchain
  - faiss-cpu
  - openai, google-generativeai, anthropic, huggingface-hub (as needed)
  - See each folder's requirements.txt for details

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # or the requirements file in each subfolder
   ```

2. **Set API Keys**
   - OpenAI: Enter your key in the app when prompted
   - Gemini: Set `GOOGLE_API_KEY` in code or as an environment variable
   - Claude: Set `CLAUDE_API_KEY` in code or as an environment variable

3. **Run the desired Streamlit app**
   ```bash
   streamlit run Chatbot_OpenAI/chatbot_openai.py
   # or
   streamlit run Gemini/chat_with_pdf_gemini_with_history.py
   # or
   streamlit run Claude_Model/rag_with_claude.py
   ```

4. **Upload your PDFs and start chatting!**

## Folder Structure
- `Chatbot_OpenAI/` — OpenAI GPT-4o chatbot
- `Gemini/` — Google Gemini-2.0-flash chatbot
- `Claude_Model/` — Anthropic Claude-3 RAG chatbot

## Notes
- Do not upload PDFs containing sensitive information
- Keep your API keys secure
- For research and educational use only, not for commercial purposes

## Contact
For questions or suggestions, please contact the author. 