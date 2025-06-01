"""
Agent responsible for research augmentation using a vector database (ChromaDB)
and LLM for contextual understanding and concept retrieval.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging 
from openai import OpenAI
import google.generativeai as genai
import google.generativeai.types as genai_types

from .agent import Agent
from .math_text import ExtractedConcept

# Import from config
from config import (
    DB_PATH, 
    DB_COLLECTION_NAME, 
    FRONTIER_AGENT_PREFERRED_MODELS, 
    FRONTIER_AGENT_LLM_MAX_TOKENS,
    TOKENIZER_MODEL_NAME, 
    LOG_LEVEL
)

class MathFrontierAgent(Agent):
    name = "Math Frontier (RAG)"
    color = Agent.BLUE

    def __init__(self, collection_name: Optional[str] = None):
        log_level_setting = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        super().__init__(log_level=log_level_setting)
        self.log("Initializing...")

        self.active_llm_client = None
        self.google_model_instance = None
        self.llm_provider_for_frontier = None
        self.model_name_for_frontier = None
        
        PROVIDER_PREFERENCE_ORDER = ["Google Gemini", "DeepSeek", "OpenAI"] 

        for provider_key in PROVIDER_PREFERENCE_ORDER:
            model_api_name = FRONTIER_AGENT_PREFERRED_MODELS.get(provider_key)
            if not model_api_name:
                continue
            try:
                if provider_key == "Google Gemini" and os.getenv("GOOGLE_API_KEY"):
                    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    self.model_name_for_frontier = model_api_name
                    self.google_model_instance = genai.GenerativeModel(self.model_name_for_frontier)
                    self.llm_provider_for_frontier = provider_key
                    self.log(f"FrontierAgent using {provider_key} LLM ({self.model_name_for_frontier}).")
                    break 
                elif provider_key == "DeepSeek" and os.getenv("DEEPSEEK_API_KEY"):
                    self.active_llm_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
                    self.model_name_for_frontier = model_api_name
                    self.llm_provider_for_frontier = provider_key
                    self.log(f"FrontierAgent using {provider_key} LLM ({self.model_name_for_frontier}).")
                    break
                elif provider_key == "OpenAI" and os.getenv("OPENAI_API_KEY"):
                    self.active_llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    self.model_name_for_frontier = model_api_name
                    self.llm_provider_for_frontier = provider_key
                    self.log(f"FrontierAgent using {provider_key} LLM ({self.model_name_for_frontier}).")
                    break
            except Exception as e_llm_init:
                self.log_error(f"Failed to initialize {provider_key} for FrontierAgent: {e_llm_init}")
        
        if not self.llm_provider_for_frontier:
            self.log_warning("No API keys found for preferred FrontierAgent LLMs or all attempts failed. LLM-dependent features for FrontierAgent may be unavailable.")

        self.embedding_model = SentenceTransformer(TOKENIZER_MODEL_NAME)
        self.log(f"Embedding model '{TOKENIZER_MODEL_NAME}' loaded.")

        self.log(f"Initializing ChromaDB at {DB_PATH}...")
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        
        final_collection_name = collection_name or DB_COLLECTION_NAME
        self.collection = chroma_client.get_or_create_collection(final_collection_name)
        self.log(f"ChromaDB collection '{final_collection_name}' ready. Count: {self.collection.count()}")

    def embed_text(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    def store_concepts(self, concepts: List[ExtractedConcept]):
        if not concepts:
            return
        ids_to_add, embeddings_to_add, documents_to_add, metadatas_to_add = [], [], [], []
        for concept in concepts:
            if not concept.content:
                self.log_warning(f"Concept {concept.concept_id} has no content. Skipping storage.")
                continue
            try:
                ids_to_add.append(concept.concept_id)
                embedding = self.embed_text(concept.content)
                embeddings_to_add.append(embedding)
                documents_to_add.append(concept.content)
                metadatas_to_add.append(concept.to_chroma_metadata())
            except Exception as e:
                self.log_error(f"Failed to process concept {concept.concept_id} for storage: {e}")
        if not ids_to_add:
            self.log("No valid concepts to store after processing.")
            return
        try:
            self.collection.upsert(ids=ids_to_add, embeddings=embeddings_to_add, documents=documents_to_add, metadatas=metadatas_to_add)
            self.log(f"Successfully stored/upserted {len(ids_to_add)} concepts.")
        except Exception as e:
            self.log_error(f"Failed to add/upsert concepts to ChromaDB: {e}")

    def find_similar_concepts(self, query_text: str, n_results: int = 5, where_clause: Optional[Dict] = None) -> List[ExtractedConcept]:
        self.log(f"Querying for similar concepts (n={n_results}): {query_text[:60]}...")
        if not query_text.strip():
            self.log_warning("Query text for similarity search is empty. Returning no results.")
            return []
        vector = self.embed_text(query_text)
        try:
            results = self.collection.query(query_embeddings=[vector], n_results=n_results, include=['documents', 'metadatas'], where=where_clause)
            found_concepts = []
            if results and results['ids'] and results['ids'][0]:
                for i, _ in enumerate(results['ids'][0]):
                    meta = results['metadatas'][0][i]
                    content = results['documents'][0][i]
                    found_concepts.append(ExtractedConcept(
                        concept_id=meta.get('concept_id', f"unknown_id_{i}"), 
                        document_id=meta.get('document_id', 'unknown_doc'),
                        concept_type=meta.get('concept_type', 'Unknown'),
                        label=meta.get('label'),
                        content=content,
                        source_file_path=meta.get('source_file_path', 'unknown_path'),
                        page_number=int(meta.get('page_number', 0)) ))
                self.log(f"Found {len(found_concepts)} similar concepts.")
            return found_concepts
        except Exception as e:
            self.log_error(f"Error querying ChromaDB for similar concepts: {e}")
            return []

    def find_concept_by_label(self, label: str, doc_id: Optional[str] = None, n_results: int = 1) -> Optional[ExtractedConcept]:
        self.log(f"Querying for label '{label}'" + (f" in doc '{doc_id}'" if doc_id else ""))
        if doc_id:
            where_clause_for_query = {"$and": [{"label": {"$eq": label}}, {"document_id": {"$eq": doc_id}}]}
        else:
            where_clause_for_query = {"label": {"$eq": label}}
        try:
            results = self.collection.query(
                query_texts=[label], 
                where=where_clause_for_query,
                n_results=n_results,
                include=['documents', 'metadatas'] )
            if results and results['ids'] and results['ids'][0]:
                meta = results['metadatas'][0][0]
                content = results['documents'][0][0]
                concept = ExtractedConcept(
                    concept_id=meta.get('concept_id', 'unknown_id'),
                    document_id=meta.get('document_id', 'unknown_doc'),
                    concept_type=meta.get('concept_type', 'Unknown'),
                    label=meta.get('label'),
                    content=content,
                    source_file_path=meta.get('source_file_path', 'unknown_path'),
                    page_number=int(meta.get('page_number', 0)) )
                self.log(f"Found concept by label: {concept.get_full_id()}")
                return concept
            self.log(f"No concept found for label '{label}'" + (f" in doc '{doc_id}'" if doc_id else ""))
            return None
        except Exception as e:
            self.log_error(f"Error getting concept by label '{label}': {e}")
            return None

    def generate_llm_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        actual_max_tokens = max_tokens if max_tokens is not None else FRONTIER_AGENT_LLM_MAX_TOKENS
        if not (self.active_llm_client or self.google_model_instance):
            self.log_error("LLM response generation failed: No LLM client is configured for FrontierAgent.")
            return "Error: No LLM provider is configured for FrontierAgent's own tasks."
        self.log(f"FrontierAgent calling LLM ({self.model_name_for_frontier} via {self.llm_provider_for_frontier}) for generic response...")
        if self.llm_provider_for_frontier == "Google Gemini" and self.google_model_instance:
            try:
                google_formatted_messages = []
                system_instruction = None
                for m in messages:
                    if m["role"] == "system": system_instruction = m["content"]
                    else: google_formatted_messages.append({"role": "user" if m["role"]=="user" else "model", "parts": [m["content"]]})
                current_google_model = self.google_model_instance
                if system_instruction:
                    current_google_model = genai.GenerativeModel(self.model_name_for_frontier, system_instruction=system_instruction)
                response = current_google_model.generate_content(
                    google_formatted_messages, 
                    generation_config=genai_types.GenerationConfig(max_output_tokens=actual_max_tokens) )
                return response.text or "Error: Empty LLM response from Google."
            except Exception as e:
                self.log_error(f"Google LLM call failed in FrontierAgent's generate_llm_response: {e}")
                return f"Error: Google LLM failed. {e}"
        elif self.active_llm_client: 
            try:
                response = self.active_llm_client.chat.completions.create(
                    model=self.model_name_for_frontier, 
                    messages=messages, 
                    seed=42,
                    max_tokens=actual_max_tokens )
                return response.choices[0].message.content or "Error: Empty LLM response."
            except Exception as e:
                self.log_error(f"LLM call ({self.llm_provider_for_frontier}) failed in FrontierAgent's generate_llm_response: {e}")
                return f"Error: LLM call failed. {e}"
        else:
            return "Error: LLM client not available for FrontierAgent."
