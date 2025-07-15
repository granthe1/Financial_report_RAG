# Claude Opus 4 RAG System for 10-K Document Analysis

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using Claude Opus 4 to analyze 10-K reports from major technology companies (Alphabet/Google, Amazon, Microsoft). The system provides accurate and detailed answers strictly based on the information contained in the 10-K documents, with all responses formatted as plain text only.

The system leverages Claude Opus 4, one of the most advanced AI models available, offering superior reasoning and analytical capabilities.

## Key Features

### RAG Implementation
- **Claude Opus 4 Integration**: Utilizes the most advanced Claude model for high-level reasoning and analysis
- **Sophisticated Chunking**: Advanced text splitting optimized for Opus 4's context understanding
- **MMR Retrieval**: Maximum Marginal Relevance for diverse and relevant document retrieval
- **Context-Aware Responses**: Maintains conversation history for improved continuity
- **Enhanced Reasoning**: Leverages Opus 4's advanced capabilities for complex financial analysis

### Data Leakage Prevention
- **Strict Evaluation Protocol**: Separate training and testing datasets
- **Controlled Testing**: Only a subset of questions is used for evaluation
- **Validation Framework**: Comprehensive evaluation metrics without overfitting

### User-Friendly Interface
- **Streamlit Web App**: Modern, responsive interface
- **Real-time Configuration**: Adjustable model parameters
- **Live Evaluation Metrics**: Real-time performance tracking
- **Export Capabilities**: Save chat history and evaluation results

## Project Structure

```
Final/
├── rag_with_claude.py              # Main RAG application
├── evaluate_rag_system.py          # Evaluation framework
├── requirements_claude.txt         # Dependencies for Claude system
├── Q&A.xlsx                        # Question dataset
├── rag_config.json                 # System configuration and prompt
├── README_CLAUDE_RAG.md            # This file
```

## Installation and Setup

### 1. Environment Setup

```bash
# Create conda environment
conda create -n claude_rag python=3.11
conda activate claude_rag

# Install dependencies
pip install -r requirements_claude.txt
```

### 2. API Configuration

1. Ensure your Claude API key is in `Claude API.txt`
2. The system will automatically load the API key

### 3. Run the Application

```bash
# Start the main RAG application
streamlit run rag_with_claude.py

# Run evaluation and prompt optimization
python evaluate_rag_system.py
```

## System Architecture

### Core Components

1. **Document Processing**
   - PDF loading with PyPDFLoader
   - Advanced text chunking (1500 chars with 300 char overlap)
   - Metadata enrichment for tracking

2. **Vector Store**
   - FAISS for efficient similarity search
   - Claude embeddings for semantic understanding
   - MMR retrieval for diverse results

3. **RAG Chain**
   - Claude Opus 4 for generation
   - Custom prompt engineering (see below)
   - Source document tracking

4. **Evaluation and Optimization**
   - Automated quality assessment using semantic similarity
   - Multiple evaluation metrics (relevance, completeness, sources used)
   - Automatic prompt optimization: tests multiple candidate prompts, saves the best to `rag_config.json`
   - Data leakage prevention

### Enhanced System Prompt

The system uses a carefully crafted prompt, now requiring:
- All answers must be in plain text only
- Do not use any Markdown formatting, bold, italics, bullet points, or code blocks under any circumstances
- Write in clear, logical, and fluent English, using only plain text and clear structure
- Make your answer easy to read
- If the answer is not in the documents, say you don't know
- If the answer involves a numerical value (such as revenue, expenses, or other metrics), only extract it from clearly presented tables or explicitly stated values, never infer or synthesize, and always default to finalized figures in financial statement tables. If conflicting values appear, use the one in the tabular financial statement section. If no such finalized figure exists, respond that the information is insufficient.

**Prompt excerpt:**
> You are a financial analyst. Answer strictly based on the provided 10-K documents. Your answer must be in plain text only. Do not use any Markdown formatting, bold, italics, bullet points, or code blocks under any circumstances. Write in clear, logical, and fluent English, using only plain text and clear structure. Make your answer easy to read. If the answer is not in the documents, say you don't know. If the answer involves a numerical value, such as revenue, expenses, or other metrics, you must extract it only from clearly presented tables or explicitly stated figures in the 10-K documents. Do not calculate or infer values from descriptive narrative text or comparative phrases such as "increased by $X billion." Do not rely on summaries or interpretations. If the information is ambiguous or not explicitly provided, respond by saying: "The information is not explicitly stated in the documents." Do not guess, do not hallucinate, and do not use any external or prior knowledge under any circumstances. Do not interpret tone, intent, or attitude. Only report what is explicitly stated in the document. Only respond if you can directly quote or reference the document. If not, say the information is insufficient. When answering with a numerical value, prioritize annual summary tables or year-end financial line items over any narrative discussion. Never synthesize or combine multiple values across the text. A valid answer must come from a single, clearly stated and finalized figure — preferably in a labeled financial statement table. If conflicting values appear in different parts of the document, you must default to the one in the tabular financial statement section. If no such finalized figure exists, you must respond that the information is insufficient.

## Evaluation Metrics

- **Relevance Score**: Semantic similarity between AI and reference answers
- **Completeness Score**: Assesses answer thoroughness
- **Source Usage**: Checks if the answer is grounded in the provided documents

## Best Practices

- Always ensure the system prompt in `rag_config.json` is up to date
- If you see formatting issues (e.g., bold or italics), clear the Streamlit cache and restart the app
- For optimal results, use high-quality, text-searchable PDFs and ask specific, focused questions

## Troubleshooting

- **Formatting issues**: The system now automatically removes Markdown special characters (*, _, `, $) from all AI answers before displaying them, ensuring no formatting artifacts (such as bold, italics, or formulas) appear in the output. Answers are rendered with `st.write()` for automatic line wrapping and improved readability.
- **API Key Errors**: Verify API key format and validity
- **Performance Issues**: Reduce chunk size or retrieval count for faster processing

## Contributing

- Follow the existing code structure
- Add comprehensive error handling
- Include unit tests for new features
- Update documentation for changes 