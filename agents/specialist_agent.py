"""
Agent that uses a Large Language Model (LLM) with a specialized prompt
to extract structured mathematical concepts and their references in JSON format
from text chunks. Supports multiple LLM providers via model identifiers.
"""
import os
import json
import re
from typing import List, Dict, Any, Optional
import logging 
from openai import OpenAI 
import google.generativeai as genai
import google.generativeai.types as genai_types
# Conditional imports for other SDKs:
# import anthropic 
# import cohere 

# Updated imports:
from .agent import Agent

# Import from config
from config import (
    MODEL_CONFIG, 
    UI_LLM_MODEL_CHOICES, 
    SPECIALIST_LLM_MAX_TOKENS, 
    SPECIALIST_LLM_TEMPERATURE,
    LOG_LEVEL
)

class MathSpecialistAgent(Agent):
    name = "Math Specialist (LLM API)"
    color = Agent.RED

    def __init__(self, llm_model_identifier: str):
        log_level_setting = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        super().__init__(log_level=log_level_setting)
        
        if llm_model_identifier not in MODEL_CONFIG:
            self.log_error(f"Unknown LLM model identifier: {llm_model_identifier}. Available: {UI_LLM_MODEL_CHOICES}")
            raise ValueError(f"Unknown LLM model identifier: {llm_model_identifier}")

        config_for_model = MODEL_CONFIG[llm_model_identifier]
        self.llm_provider = config_for_model["provider"]
        self.model_name = config_for_model["api_model_name"] 
        self.ui_model_identifier = llm_model_identifier 

        self.log(f"Initializing LLM client for provider: {self.llm_provider}, model: {self.model_name} (UI selection: {self.ui_model_identifier})")

        self.client = None 
        self.anthropic_client = None 
        self.cohere_client = None    

        try:
            if self.llm_provider == "OpenAI":
                api_key = os.getenv(config_for_model.get("api_key_env") or "OPENAI_API_KEY")
                if not api_key: raise ValueError("OpenAI API Key not found.")
                self.client = OpenAI(api_key=api_key)
            elif self.llm_provider == "Google Gemini":
                api_key = os.getenv(config_for_model.get("api_key_env") or "GOOGLE_API_KEY")
                if not api_key: raise ValueError("Google API Key not found.")
                genai.configure(api_key=api_key)
            elif self.llm_provider == "DeepSeek":
                api_key = os.getenv(config_for_model.get("api_key_env") or "DEEPSEEK_API_KEY")
                base_url = config_for_model.get("base_url") 
                if not api_key: raise ValueError("DeepSeek API Key not found.")
                if not base_url: raise ValueError("DeepSeek base_url not configured.")
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            elif self.llm_provider == "Anthropic":
                import anthropic 
                api_key = os.getenv(config_for_model.get("api_key_env") or "ANTHROPIC_API_KEY")
                if not api_key: raise ValueError("Anthropic API Key not found.")
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            elif self.llm_provider == "Cohere":
                import cohere 
                api_key = os.getenv(config_for_model.get("api_key_env") or "COHERE_API_KEY")
                if not api_key: raise ValueError("Cohere API Key not found.")
                self.cohere_client = cohere.Client(api_key)
            elif self.llm_provider in ["Mistral", "Meta (Llama)", "Qwen (Alibaba)", "Microsoft (Phi)"]:
                api_key_env_var = config_for_model.get("api_key_env")
                api_key = os.getenv(api_key_env_var) if api_key_env_var else None
                base_url = config_for_model.get("base_url")
                if not api_key: raise ValueError(f"{api_key_env_var or 'API key for ' + self.llm_provider} not found.")
                if not base_url: raise ValueError(f"Base URL not configured for {self.llm_provider} in MODEL_CONFIG.")
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                raise ValueError(f"Unsupported LLM provider derived: {self.llm_provider}")
            self.log(f"SpecialistAgent configured for {self.llm_provider} using API model {self.model_name}.")
        except ValueError as e:
            self.log_error(f"Configuration error for {self.llm_provider} ({self.ui_model_identifier}): {e}")
            raise
        except ImportError as e:
            self.log_error(f"Missing SDK for {self.llm_provider}: {e}. Please install the required package.")
            raise
        except Exception as e:
            self.log_error(f"Failed to initialize LLM client for {self.llm_provider} ({self.ui_model_identifier}): {e}")
            raise

    def _construct_extraction_prompt_system(self) -> str:
        return """
You are an expert AI assistant specializing in mathematical text analysis. Your task is to extract key mathematical concepts and any references within them from the provided text chunk.
For each distinct concept you identify, provide the following information in a JSON object:
1.  `concept_type`: Choose the *most specific and appropriate* type from this list: [Definition, Theorem, Lemma, Corollary, Proposition, Proof, Equation, Remark, Example, Axiom, Conjecture, Algorithm, Claim, Identity, Notation, Problem, Solution, Method, Assumption, Observation, Principle, Fact, Note, Exercise, Question, Answer, Derivation, Procedure, Result, FigureCaption, TableCaption, Abstract, Introduction, Conclusion, SectionHeader, SubsectionHeader, Paragraph, Motivation, Background, Methodology].
2.  `label`: The formal label if any (e.g., "Theorem 3.1", "Definition 2", "(5.1)", "Eq. 3"). If no explicit label exists, use `null`.
3.  `content`: The exact text or LaTeX mathematical expression of the concept. This string MUST be a valid JSON string value. This means all special characters, especially backslashes (`\\`) from LaTeX (e.g., in `\\ldots` or `\\frac`), must be correctly escaped (e.g., as `\\\\ldots` or `\\\\frac`). Ensure Unicode characters are also correctly represented (e.g., `\\uXXXX`).
4.  `references`: A list of JSON objects, where each object represents a reference made *within the content of this specific concept*. Each reference object should have:
    * `raw_text`: The snippet of text that contains the reference (e.g., "see Theorem 1", "as shown in [12, Section 2]", "refer to eq. (3)").
    * `target_document_hint`: A hint for the target document if the reference is external (e.g., "[12]", "[Smith2023]", "paper [X]"). Use `null` if the reference is internal to the current document or if no such hint is present.
    * `target_label_hint`: A hint for the target label within a document (e.g., "Theorem 1", "Section 2", "eq. (3)", "Chapter 5"). Use `null` if no specific label is mentioned.

Output ONLY a valid JSON list `[...]` of these concept objects. If no concepts are found in the text chunk, output an empty list `[]`.
Ensure your entire output is a single, valid JSON array. Do not include any explanatory text before or after the JSON array.
"""
    def _construct_extraction_prompt_user(self, text_chunk: str, context: Optional[str]) -> str:
        context_prompt = f"Consider this related context from the document (do not extract concepts from this context, only use it for understanding):\n{context}\n\n---\n\n" if context else ""
        return f"{context_prompt}Analyze the following text chunk and extract concepts:\n--- TEXT CHUNK ---\n{text_chunk}\n--- END TEXT CHUNK ---"

    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        self.log(f"Parsing LLM response (length: {len(response_text)}). Preview: {response_text[:100]}...")
        if not response_text or not response_text.strip():
            self.log_warning("LLM response was empty or whitespace only.")
            return []
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[len("```json"):].strip()
        if cleaned_text.startswith("```"): cleaned_text = cleaned_text[len("```"):].strip()
        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-len("```")].strip()
        if not (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
            self.log_error(f"LLM response is not a JSON list. Snippet: {cleaned_text[:300]}...")
            start_brace = cleaned_text.find('[')
            end_brace = cleaned_text.rfind(']')
            if start_brace != -1 and end_brace != -1 and start_brace < end_brace:
                self.log_warning("Attempting to extract JSON list from malformed response.")
                cleaned_text = cleaned_text[start_brace : end_brace + 1]
            else:
                self.log_error("Could not find JSON list markers '[...]' in malformed response.")
                return []
        try:
            parsed_json = json.loads(cleaned_text)
            if not isinstance(parsed_json, list):
                self.log_error(f"Parsed JSON is not a list, but type: {type(parsed_json)}. Content: {str(parsed_json)[:300]}")
                return []
            valid_items = [item for item in parsed_json if isinstance(item, dict) and "concept_type" in item and "content" in item]
            if len(valid_items) != len(parsed_json): self.log_warning("Some items in parsed JSON list were invalid and skipped.")
            return valid_items
        except json.JSONDecodeError as e:
            self.log_error(f"JSON Decode Error: {e}. Response text (cleaned snippet): {cleaned_text[:500]}...")
            return []
        except Exception as e_parse: 
            self.log_error(f"Unexpected error parsing LLM JSON response: {e_parse}. Snippet: {cleaned_text[:300]}")
            return []

    def extract_structured_info(self, text_chunk: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        if not text_chunk.strip():
            self.log_warning("Text chunk for extraction is empty. Skipping LLM call.")
            return []
        self.log(f"Requesting LLM extraction via {self.llm_provider} (API model: {self.model_name}) for text chunk (length {len(text_chunk)}): {text_chunk[:60]}...")
        system_prompt_content = self._construct_extraction_prompt_system()
        user_prompt_content = self._construct_extraction_prompt_user(text_chunk, context)
        llm_response_text = ""
        try:
            if self.llm_provider == "Google Gemini":
                model_for_call = genai.GenerativeModel(self.model_name, system_instruction=system_prompt_content)
                generation_config = genai_types.GenerationConfig(
                    temperature=SPECIALIST_LLM_TEMPERATURE, 
                    max_output_tokens=SPECIALIST_LLM_MAX_TOKENS, 
                    response_mime_type="application/json", )
                response = model_for_call.generate_content([user_prompt_content], generation_config=generation_config)
                llm_response_text = response.text
            elif self.client: 
                messages = [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}]
                use_json_mode = self.llm_provider == "OpenAI" and any(m in self.model_name for m in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125"])
                response_params = {"model": self.model_name, "messages": messages, "seed": 42, 
                                   "max_tokens": SPECIALIST_LLM_MAX_TOKENS, 
                                   "temperature": SPECIALIST_LLM_TEMPERATURE}
                if use_json_mode: response_params["response_format"] = {"type": "json_object"}
                completion = self.client.chat.completions.create(**response_params)
                llm_response_text = completion.choices[0].message.content
            elif self.llm_provider == "Anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model=self.model_name, max_tokens=SPECIALIST_LLM_MAX_TOKENS, 
                    temperature=SPECIALIST_LLM_TEMPERATURE, system=system_prompt_content, 
                    messages=[{"role": "user", "content": user_prompt_content}] )
                if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                    llm_response_text = response.content[0].text
                else: self.log_warning("Anthropic response format not as expected."); llm_response_text = ""
            elif self.llm_provider == "Cohere" and self.cohere_client:
                response = self.cohere_client.chat(
                    model=self.model_name, message=user_prompt_content, preamble=system_prompt_content, 
                    temperature=SPECIALIST_LLM_TEMPERATURE, max_tokens=SPECIALIST_LLM_MAX_TOKENS, )
                llm_response_text = response.text
            else:
                self.log_error(f"LLM provider {self.llm_provider} not configured for API call or client not available."); return []
            if not llm_response_text:
                self.log_warning(f"LLM ({self.llm_provider} - {self.model_name}) returned an empty response string."); return []
            return self._parse_llm_response(llm_response_text)
        except genai_types.BlockedPromptException as e_blocked_gemini:
            self.log_error(f"Google Gemini API call failed due to blocked prompt: {e_blocked_gemini}"); return []
        except ConnectionError as e_conn: 
            self.log_error(f"LLM connection error for {self.llm_provider}: {e_conn}"); return []
        except Exception as e_api: 
            self.log_error(f"LLM API call failed ({self.llm_provider}, {self.model_name}): {e_api}")
            if hasattr(e_api, 'response') and e_api.response: self.log_error(f"API Response (if available): {getattr(e_api.response, 'text', 'N/A')}")
            return []
