# Polymath Parser: Mathematical PDF to Structured JSON Converter

---

## Overview

Polymath Parser is an agent-based system designed to process mathematical PDF documents. It extracts key concepts (such as definitions, theorems, equations) and converts them into a structured JSON format. The system leverages various Large Language Models (LLMs) for nuanced understanding and extraction, and uses a vector database for research augmentation and concept linking.

A Gradio-based web interface is provided for easy interaction, including PDF upload, model selection, and viewing of results.

---

## Features

* **PDF Processing:** Converts PDF documents into processable text, utilizing the Mathpix API for high-quality Markdown conversion of mathematical content.
* **Multi-Agent Architecture:** Employs specialized agents for different tasks:
    * `PDFProcessingAgent`: Handles PDF-to-text/Markdown conversion and metadata extraction.
    * `MathTextProcessor`: Cleans and chunks text for LLM processing.
    * `SpecialistAgent`: Uses a selected LLM to extract structured mathematical concepts and references.
    * `FrontierAgent`: Manages a ChromaDB vector store for concept embeddings, enabling semantic search and RAG.
    * `ReferencingAgent`: Resolves internal and external (bibliographic) references.
    * `KeywordExtractionAgent`: Identifies potential labels and citation markers.
    * `EnsembleAgent`: Coordinates specialist agents.
    * `AnalysisPlanner`: Orchestrates the overall workflow.
* **LLM Integration:** Supports multiple LLM providers (OpenAI, Google Gemini, Anthropic, Cohere, DeepSeek, and other OpenAI-compatible endpoints). Model choice is configurable via the UI.
* **Vector Database:** Stores extracted concepts and embeddings (via Sentence-Transformers) in ChromaDB.
* **Structured Output:** Generates detailed JSON files for each processed PDF.
* **Web Interface:** User-friendly Gradio UI for all major interactions.
* **Configuration Management:** Centralized settings in `config.py` and API key management via `.env`.
* **Caching:** Caches Mathpix Markdown conversions.

---

## Architecture

The system uses a collection of cooperative agents for its PDF processing and analysis pipeline:

1.  **UI Interaction (`polymath_parser_ui.py`):** User uploads PDFs and selects settings.
2.  **Framework Orchestration (`app/polymath_parser_framework.py`):** Initializes agents and manages the process.
3.  **Planning (`agents/planning_agent.py`):** The `AnalysisPlanner` coordinates the analysis.
4.  **PDF Processing (`agents/pdf_processing_agent.py`):** Converts PDF and extracts metadata/bibliography.
5.  **Text Preparation (`processing/math_text_processor.py`):** Scrubs and chunks text.
6.  **Core Analysis (coordinated by `agents/ensemble_agent.py`):**
    * **RAG Context (`agents/frontier_agent.py`):** Provides context from the vector store.
    * **Concept Extraction (`agents/specialist_agent.py`):** LLM extracts concepts and references.
    * **Reference Resolution (`agents/referencing_agent.py`):** Links references.
7.  **Storage (`agents/frontier_agent.py`):** New concepts are stored in ChromaDB.
8.  **Output:** Results are saved as JSON and displayed in the UI.

---

## Setup and Installation

### Prerequisites

* Python 3.9+
* `pip` and `virtualenv` (recommended)

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd polymath-parser
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Or, if you modify `requirements.in`: `pip install pip-tools && pip-compile requirements.in -o requirements.txt && pip install -r requirements.txt`)

4.  **Set Up API Keys (.env file):**
    Create a `.env` file in the project root. Add necessary keys:
    ```env
    # Mathpix API (Optional)
    MATHPIX_APP_ID="your_mathpix_app_id"
    MATHPIX_APP_KEY="your_mathpix_app_key"

    # LLM API Keys
    OPENAI_API_KEY="sk-your_openai_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    # ... add other keys (ANTHROPIC_API_KEY, COHERE_API_KEY, etc.)

    # For OpenAI-compatible models (if applicable)
    MISTRAL_API_BASE="your_mistral_provider_base_url"
    MISTRAL_API_KEY="your_mistral_provider_api_key"
    # ... other similar provider settings
    ```

---

## Configuration

Global configurations are managed in `config.py`. This includes paths, logging levels, text processing parameters, vector database settings, LLM model definitions, and UI settings. API keys are loaded from the `.env` file.

---

## How to Run

Launch the Gradio web interface:

```bash
python app/polymath_parser_ui.py