Polymath Parser: Mathematical PDF to Structured JSON Converter
Overview
Polymath Parser is an agent-based system designed to process mathematical PDF documents, extract key concepts (such as definitions, theorems, equations), and convert them into a structured JSON format. It leverages various Large Language Models (LLMs) for nuanced understanding and extraction, and uses a vector database for research augmentation and concept linking. The system provides a Gradio-based web interface for easy interaction, PDF upload, model selection, and viewing of results.

Features
PDF Processing: Converts PDF documents into processable text, utilizing Mathpix API for high-quality Markdown conversion of mathematical content.

Multi-Agent Architecture: Employs specialized agents for different tasks:

PDFProcessingAgent: Handles PDF-to-text/Markdown conversion and metadata extraction.

MathTextProcessor: Cleans and chunks text for LLM processing.

SpecialistAgent: Uses a selected LLM to extract structured mathematical concepts and references from text chunks.

FrontierAgent: Manages a ChromaDB vector store for concept embeddings, enabling semantic search and Retrieval Augmented Generation (RAG).

ReferencingAgent: Resolves internal and external (bibliographic) references within extracted concepts.

KeywordExtractionAgent: Identifies potential labels and citation markers.

EnsembleAgent: Coordinates specialist agents to analyze document chunks.

AnalysisPlanner: Orchestrates the overall workflow for a document.

LLM Integration: Supports multiple LLM providers (OpenAI, Google Gemini, Anthropic, Cohere, DeepSeek, and other OpenAI-compatible endpoints like Mistral, Llama) for concept extraction. Model choice is configurable via the UI.

Vector Database: Stores extracted concepts and their embeddings (using Sentence-Transformers) in ChromaDB for similarity searches and contextual understanding.

Structured Output: Generates detailed JSON files for each processed PDF, containing document metadata, extracted concepts, and resolved references.

Web Interface: User-friendly Gradio UI for:

Uploading PDF files.

Selecting the LLM model for analysis.

Specifying an output directory for results.

Viewing real-time logs.

Browsing processed documents and their extracted concepts.

Configuration Management: Centralized configuration (config.py) for paths, model parameters, and operational thresholds. API keys are managed via a .env file.

Caching: Caches Mathpix Markdown conversions to save on API calls and processing time.

Architecture
The system is built around a collection of cooperative agents, each responsible for a specific part of the PDF processing and analysis pipeline.

UI Interaction (polymath_parser_ui.py): User uploads PDFs and selects settings.

Framework Orchestration (polymath_parser_framework.py): Initializes agents and manages the overall process.

Planning (planning_agent.py): The AnalysisPlanner receives PDF paths and coordinates the analysis.

PDF Processing (pdf_processing_agent.py): Converts PDF to text/Markdown and extracts basic metadata and bibliography.

Text Preparation (math_text_processor.py): Scrubs and chunks the text.

Core Analysis (coordinated by ensemble_agent.py):

RAG Context (frontier_agent.py): Finds similar concepts from the vector store to provide context.

Concept Extraction (specialist_agent.py): An LLM extracts structured concepts (definitions, theorems, etc.) and references from text chunks, guided by the RAG context.

Reference Resolution (referencing_agent.py): Attempts to link identified references to other concepts within the document or to bibliographic entries.

Storage (frontier_agent.py): Newly extracted concepts and their embeddings are stored in the ChromaDB vector database.

Output: Results are saved as JSON files and displayed in the UI.

Setup and Installation
Prerequisites
Python 3.9+ (Python 3.11 was used for generating requirements.txt via pip-compile)

pip and virtualenv (recommended)

Installation Steps
Clone the Repository:

git clone <your-repository-url>
cd polymath-parser 

Create and Activate a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
The project uses pip-tools for managing dependencies.

If you want to install exact versions from the pre-compiled requirements.txt:

pip install -r requirements.txt

If you want to re-compile dependencies from requirements.in (e.g., after modifying it):

pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip install -r requirements.txt

The requirements.in file lists direct dependencies, and pip-compile generates requirements.txt with all pinned versions.

Set Up API Keys (.env file):
Create a .env file in the project root directory. This file will store your API keys. Add the following keys that you intend to use:

# Mathpix API (Optional, for PDF to Markdown conversion)
MATHPIX_APP_ID="your_mathpix_app_id"
MATHPIX_APP_KEY="your_mathpix_app_key"

# LLM API Keys (add all you plan to use)
OPENAI_API_KEY="sk-your_openai_api_key"
GOOGLE_API_KEY="your_google_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
COHERE_API_KEY="your_cohere_api_key"
DEEPSEEK_API_KEY="your_deepseek_api_key"

# For other OpenAI-compatible models (e.g., Mistral, Llama via providers)
# These are referenced by 'base_url_env' and 'api_key_env' in config.py
MISTRAL_API_BASE="your_mistral_provider_base_url" 
MISTRAL_API_KEY="your_mistral_provider_api_key"

LLAMA_API_BASE="your_llama_provider_base_url" 
LLAMA_API_KEY="your_llama_provider_api_key"

# Add other keys/base URLs as needed (e.g., QWEN_API_BASE, PHI_API_BASE)

The application will load these variables at runtime.

Configuration
Global configurations for the application are managed in config.py. This includes:

Default paths for output and caching.

Logging levels.

Parameters for text processing (tokenizer model, chunk sizes).

Vector database settings (path, collection name).

LLM model definitions for the UI (MODEL_CONFIG), including provider mapping, API model names, and optional base URLs or specific API key environment variable names.

Default parameters for LLM calls (max tokens, temperature).

UI display settings.

You can modify config.py to change these default behaviors.

How to Run
Once the setup is complete, you can launch the Gradio web interface:

python polymath_parser_ui.py

This will start a local web server, and the UI address (usually http://127.0.0.1:7860 or http://0.0.0.0:7860) will be printed to the console. Open this address in your web browser.

Using the UI:

Upload PDFs: Use the file upload component to select one or more PDF files.

Select LLM Model: Choose your preferred LLM from the dropdown menu. Ensure you have the corresponding API key in your .env file and that the model is correctly configured in config.py.

Output Directory (Optional): Specify a directory to save the output JSON files. If left blank, it defaults to the directory specified in config.py (math_json_output by default, created in the project root).

Start Analysis: Click the "Start Analysis" button.

Monitor Progress: View real-time logs in the "Real-time Logs" section.

View Results:

The "Processed Documents" table will list the documents analyzed in the current session or loaded from the output directory.

Click on a document in the table to see its extracted concepts in the "Extracted Concepts Detail" section.

The "Load/Refresh Results from Output Directory" button can be used to load previously processed JSON files from the specified output directory into the UI.

Directory Structure
polymath-parser/
├── agents/                  # Core agent logic
│   ├── __init__.py
│   ├── agent.py             # Base agent class
│   ├── ensemble_agent.py
│   ├── frontier_agent.py    # RAG and vector DB interaction
│   ├── keyword_extraction_agent.py
│   ├── math_text.py         # Pydantic models for data structures
│   ├── pdf_processing_agent.py
│   ├── planning_agent.py
│   ├── referencing_agent.py
│   └── specialist_agent.py  # LLM-based concept extraction
├── .polymath_parser_cache/  # Default cache directory (created automatically)
│   └── md_cache/            # Cached Mathpix Markdown files
├── math_json_output/        # Default output directory for JSON results (created automatically)
├── math_vectorstore/        # Default directory for ChromaDB (created automatically)
├── .env                     # API keys and environment variables (you need to create this)
├── config.py                # Centralized application configuration
├── log_utils.py             # Utilities for log formatting
├── math_text_processor.py   # Text cleaning, chunking, bibliography extraction
├── polymath_parser_framework.py # Main application orchestrator
├── polymath_parser_ui.py    # Gradio web interface
├── requirements.in          # List of direct dependencies for pip-compile
├── requirements.txt         # Pinned versions of all dependencies
├── LICENSE                  # Project license file (MIT License)
└── README.md                # This file

Key Technologies & Libraries
Python: Core programming language.

Gradio: For creating the web UI.

Large Language Models (LLMs): OpenAI GPT models, Google Gemini, Anthropic Claude, Cohere models, DeepSeek, etc., via their respective SDKs or OpenAI-compatible APIs.

Sentence-Transformers: For generating text embeddings.

ChromaDB: Vector database for storing and querying concept embeddings.

Pydantic: For data validation and modeling.

PyMuPDF (fitz): For PDF text and metadata extraction.

Requests: For HTTP requests (e.g., to Mathpix API).

Mathpix API (Optional): For converting PDFs with mathematical content to Markdown.

python-dotenv: For managing environment variables (API keys).

pip-tools: For managing Python package dependencies.

Potential Future Enhancements
Improved page number tracking for extracted concepts.

More sophisticated bibliography parsing for various formats.

Enhanced internal reference resolution across multiple documents.

Support for other input formats besides PDF.

Advanced visualization of concept relationships.

Batch processing mode without UI for command-line execution.

More granular error handling and reporting in the UI.

Option to re-process a document with a different LLM without re-uploading.

License
This project is licensed under the MIT License - see the LICENSE file for details.