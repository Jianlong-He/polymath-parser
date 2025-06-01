"""
Utility class for cleaning, tokenizing, and preparing text extracted from 
mathematical documents, handling both raw text and LaTeX-like content.
"""
from typing import List, Dict
import re
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Import from config
from config import TOKENIZER_MODEL_NAME, CHUNK_MAX_TOKENS, MIN_CHUNK_LENGTH_CHARS

class MathTextProcessor:
    """
    Provides methods for text scrubbing, bibliography extraction, and text chunking
    tailored for mathematical documents.
    """
    LATEX_REMOVAL_PATTERNS = [
        r'\\documentclass(?:\[.*?\])?\{.*?\}', r'\\usepackage(?:\[.*?\])?\{.*?\}',
        r'\\(?:sub)?section\*?\{.*?\}', r'\\(?:sub)?paragraph\*?\{.*?\}',
        r'\\author\{.*?\}', r'\\title\{.*?\}', r'\\date\{.*?\}', r'\\maketitle', r'\\thanks\{.*?\}',
        r'\\begin\{(?:document|abstract|keywords|figure|table|center|itemize|enumerate|lstlisting|verbatim|tikzpicture)\}',
        r'\\end\{(?:document|abstract|keywords|figure|table|center|itemize|enumerate|lstlisting|verbatim|tikzpicture)\}',
        r'\\includegraphics(?:\[.*?\])?\{.*?\}', r'\\caption\{.*?\}', r'\\label\{.*?\}', r'\\ref\{.*?\}', r'\\cite\{.*?\}',
        r'\\footnote\{.*?\}', r'\\pagestyle\{.*?\}', r'\\thispagestyle\{.*?\}', r'\\graphicspath\{.*?\}',
        r'\\ ऊपर$', 
        r'\^(?:Page \d+|\d+)$', 
    ]
    REPLACEMENT_PATTERNS = {
        r'\\(?:newline|par|break|smallbreak|medbreak|bigbreak|\\\\)': ' ', 
        r'\\(?:left|right)([\[\](){}.,|])': r'\1', 
        r'\s*\n\s*': '\n', 
        r'[ \t]+': ' ',    
    }
    BIBLIOGRAPHY_START_PATTERNS = [
        r'\\(?:begin\{thebibliography\}|section\*?\{References\}|chapter\*?\{Bibliography\})',
        r'\nReferences\n', r'\nBibliography\n', 
    ]
    BIB_ITEM_LATEX_PATTERN = r'\\bibitem\{(.*?)\}([\s\S]*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)'

    def __init__(self):
        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
            print(f"MathTextProcessor: Tokenizer '{TOKENIZER_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load tokenizer '{TOKENIZER_MODEL_NAME}'. MathTextProcessor is non-functional. Error: {e}")
            raise 

    def scrub_text(self, text: str) -> str:
        if not text: return ""
        scrubbed = text
        for pattern in self.LATEX_REMOVAL_PATTERNS:
            scrubbed = re.sub(pattern, '', scrubbed, flags=re.MULTILINE | re.IGNORECASE)
        for pattern, replacement in self.REPLACEMENT_PATTERNS.items():
            scrubbed = re.sub(pattern, replacement, scrubbed)
        scrubbed = re.sub(r'\n\s*\n', '\n\n', scrubbed) 
        scrubbed = scrubbed.replace('\n', ' ') 
        scrubbed = re.sub(r'[ \t]+', ' ', scrubbed) 
        return scrubbed.strip()

    def extract_bibliography(self, text_content: str) -> Dict[str, str]:
        bibliography_section_text = None
        for start_pattern_regex in self.BIBLIOGRAPHY_START_PATTERNS:
            match = re.search(start_pattern_regex + r'([\s\S]*)', text_content, re.IGNORECASE)
            if match:
                potential_bib_text = match.group(1)
                end_bib_match = re.search(r'\\end\{thebibliography\}|\\(?:section|chapter)\*?\{', potential_bib_text, re.IGNORECASE)
                if end_bib_match: bibliography_section_text = potential_bib_text[:end_bib_match.start()]
                else: bibliography_section_text = potential_bib_text 
                break
        if not bibliography_section_text: return {}
        extracted_bib_items = {}
        latex_items = re.findall(self.BIB_ITEM_LATEX_PATTERN, bibliography_section_text)
        for key, content in latex_items:
            extracted_bib_items[f"[{key.strip()}]"] = self.scrub_text(content.strip()) 
        if extracted_bib_items: print(f"MathTextProcessor: Extracted {len(extracted_bib_items)} bibliography items.")
        return extracted_bib_items

    def chunk_text(self, text: str) -> List[str]:
        if not text or not text.strip(): return []
        paragraphs = text.split('\n\n') 
        final_chunks = []
        current_chunk_token_ids: List[int] = []
        for para_text in paragraphs:
            if not para_text.strip(): continue
            para_token_ids = self.tokenizer.encode(para_text, add_special_tokens=False)
            if len(current_chunk_token_ids) + len(para_token_ids) > CHUNK_MAX_TOKENS and current_chunk_token_ids:
                chunk_str = self.tokenizer.decode(current_chunk_token_ids, skip_special_tokens=True).strip()
                if len(chunk_str) >= MIN_CHUNK_LENGTH_CHARS: final_chunks.append(chunk_str)
                current_chunk_token_ids = [] 
            if len(para_token_ids) > CHUNK_MAX_TOKENS:
                if current_chunk_token_ids:
                    chunk_str = self.tokenizer.decode(current_chunk_token_ids, skip_special_tokens=True).strip()
                    if len(chunk_str) >= MIN_CHUNK_LENGTH_CHARS: final_chunks.append(chunk_str)
                    current_chunk_token_ids = []
                for i in range(0, len(para_token_ids), CHUNK_MAX_TOKENS):
                    sub_chunk_token_ids = para_token_ids[i : i + CHUNK_MAX_TOKENS]
                    sub_chunk_str = self.tokenizer.decode(sub_chunk_token_ids, skip_special_tokens=True).strip()
                    if len(sub_chunk_str) >= MIN_CHUNK_LENGTH_CHARS: final_chunks.append(sub_chunk_str)
            else:
                if current_chunk_token_ids: current_chunk_token_ids.extend(self.tokenizer.encode('\n\n', add_special_tokens=False)) 
                current_chunk_token_ids.extend(para_token_ids)
        if current_chunk_token_ids:
            last_chunk_str = self.tokenizer.decode(current_chunk_token_ids, skip_special_tokens=True).strip()
            if len(last_chunk_str) >= MIN_CHUNK_LENGTH_CHARS: final_chunks.append(last_chunk_str)
        print(f"MathTextProcessor: Chunked text into {len(final_chunks)} chunks.")
        return final_chunks

    def get_token_count(self, text: str) -> int:
        if not text: return 0
        return len(self.tokenizer.encode(text))
