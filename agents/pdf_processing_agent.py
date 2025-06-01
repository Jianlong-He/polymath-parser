"""
Agent responsible for finding PDF files (less relevant now for UI flow), 
converting them to a processable format (Markdown via Mathpix),
extracting raw text and metadata, and preparing a ScannedPDF object.
Mathpix MD Caching is now relative to a base app data directory or a user-defined output.
"""
import os
import fitz  # PyMuPDF
from typing import List, Optional, Dict
import logging
import requests
import json
import time

from .agent import Agent
from .math_text import ScannedPDF

# Import MathTextProcessor from the 'processing' package
from processing import MathTextProcessor

# Import from config
from config import DEFAULT_APP_DATA_DIR, MD_CACHE_SUBDIR, LOG_LEVEL


class PDFProcessingAgent(Agent):
    name = "PDF Processing"
    color = Agent.WHITE

    def __init__(self, base_cache_dir: Optional[str] = None):
        log_level_setting = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        super().__init__(log_level=log_level_setting)
        
        self.app_base_cache_dir = base_cache_dir or DEFAULT_APP_DATA_DIR
        self.md_cache_dir = os.path.join(self.app_base_cache_dir, MD_CACHE_SUBDIR)
        
        os.makedirs(self.app_base_cache_dir, exist_ok=True)
        os.makedirs(self.md_cache_dir, exist_ok=True)
        
        self.log(f"MD cache directory set to: {os.path.abspath(self.md_cache_dir)}")
        
        self.text_processor = MathTextProcessor() # From processing package

        self.mathpix_app_id = os.getenv("MATHPIX_APP_ID")
        self.mathpix_app_key = os.getenv("MATHPIX_APP_KEY")

        if self.mathpix_app_id and self.mathpix_app_key:
            self.log("Mathpix API credentials loaded.")
        else:
            self.log_warning("Mathpix App ID or Key not found. Mathpix PDF-to-MD conversion will be skipped.")

    def _convert_pdf_to_mathpix_md(self, pdf_path: str) -> Optional[str]:
        if not (self.mathpix_app_id and self.mathpix_app_key):
            self.log_warning(f"Mathpix API not configured. Skipping MD conversion for {os.path.basename(pdf_path)}.")
            return None
        self.log(f"Attempting Mathpix MD conversion for {os.path.basename(pdf_path)}...")
        headers = {"app_id": self.mathpix_app_id, "app_key": self.mathpix_app_key}
        options = {
            "conversion_formats": {"md": True, "tex.zip": False, "docx": False},
            "math_inline_delimiters": ["$", "$"],
            "math_display_delimiters": ["$$", "$$"], }
        form_data = {"options_json": json.dumps(options)}
        try:
            with open(pdf_path, "rb") as f_pdf:
                files_payload = {"file": (os.path.basename(pdf_path), f_pdf)}
                response = requests.post("https://api.mathpix.com/v3/pdf",
                                         headers=headers, files=files_payload,
                                         data=form_data, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            pdf_id = response_json.get("pdf_id")
            if not pdf_id:
                self.log_error(f"Mathpix API did not return a pdf_id for {os.path.basename(pdf_path)}. Response: {response.text}")
                return None
            self.log(f"PDF {os.path.basename(pdf_path)} submitted to Mathpix. ID: {pdf_id}")
            max_polls, poll_interval = 30, 10
            for _ in range(max_polls):
                status_response = requests.get(f"https://api.mathpix.com/v3/pdf/{pdf_id}", headers=headers, timeout=30)
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get("status")
                percent_done = status_data.get("percent_done", 0)
                self.log(f"Mathpix status for {pdf_id} ({os.path.basename(pdf_path)}): {status} ({percent_done}%)")
                if status == "completed":
                    md_response = requests.get(f"https://api.mathpix.com/v3/pdf/{pdf_id}.md", headers=headers, timeout=60)
                    md_response.raise_for_status()
                    md_content = md_response.text
                    if md_content and md_content.strip():
                        self.log(f"Successfully converted {os.path.basename(pdf_path)} to MD.")
                        base_filename_pdf = os.path.splitext(os.path.basename(pdf_path))[0]
                        pdf_path_hash = hex(abs(hash(pdf_path)))[2:10] 
                        md_filename = f"{base_filename_pdf}_{pdf_path_hash}.md"
                        md_filepath = os.path.join(self.md_cache_dir, md_filename)
                        try:
                            with open(md_filepath, "w", encoding="utf-8") as f_md:
                                f_md.write(md_content)
                            self.log(f"Saved Mathpix MD to: {md_filepath}")
                        except IOError as e_save:
                            self.log_error(f"Failed to save Mathpix MD for {pdf_path} to {md_filepath}: {e_save}")
                        return md_content
                    else:
                        self.log_warning(f"Mathpix returned empty MD content for {pdf_path} (ID: {pdf_id}).")
                        return None
                elif status == "error":
                    error_msg = status_data.get('error_info', {}).get('message', status_data.get('error', 'Unknown Mathpix error'))
                    self.log_error(f"Mathpix processing failed for {pdf_id} ({os.path.basename(pdf_path)}): {error_msg}")
                    return None
                time.sleep(poll_interval)
            self.log_warning(f"Mathpix conversion timed out for {pdf_id} ({os.path.basename(pdf_path)}).")
            return None
        except requests.exceptions.RequestException as e_req:
            self.log_error(f"Mathpix API request error for {pdf_path}: {e_req}")
            if hasattr(e_req, 'response') and e_req.response is not None:
                self.log_error(f"Mathpix API Response Text: {e_req.response.text}")
            return None
        except Exception as e_generic:
            self.log_error(f"Unexpected error during Mathpix MD conversion for {pdf_path}: {e_generic}")
            return None

    def scan_for_new_pdfs(self, directory_to_scan: str, processed_doc_paths: List[str]) -> List[str]:
        self.log(f"Scanning {directory_to_scan} for new PDFs...")
        abs_scan_path = os.path.abspath(directory_to_scan)
        found_new_pdfs = []
        try:
            for filename in os.listdir(abs_scan_path):
                if filename.lower().endswith('.pdf'):
                    full_file_path = os.path.join(abs_scan_path, filename)
                    if full_file_path not in processed_doc_paths:
                        found_new_pdfs.append(full_file_path)
                        self.log(f"Identified new PDF: {filename}")
        except FileNotFoundError:
            self.log_error(f"Directory not found for scanning: {abs_scan_path}")
            return []
        except Exception as e_scan:
            self.log_error(f"Error listing or accessing directory {abs_scan_path}: {e_scan}")
            return []
        self.log(f"Found {len(found_new_pdfs)} new PDFs in {directory_to_scan}.")
        return found_new_pdfs

    def process_pdf(self, pdf_path: str) -> Optional[ScannedPDF]:
        self.log(f"Processing PDF: {pdf_path}")
        original_filename = os.path.basename(pdf_path)
        doc_id_hash_input = pdf_path 
        doc_id_hash = abs(hash(doc_id_hash_input)) % 1000000
        doc_id = f"doc_{os.path.splitext(original_filename)[0]}_{doc_id_hash:06}"
        base_filename_pdf = os.path.splitext(original_filename)[0]
        pdf_path_hash_for_cache = hex(abs(hash(pdf_path)))[2:10]
        md_filename_cache = f"{base_filename_pdf}_{pdf_path_hash_for_cache}.md"
        md_filepath_cache = os.path.join(self.md_cache_dir, md_filename_cache)
        final_md_content: Optional[str] = None
        md_available = False
        if os.path.exists(md_filepath_cache):
            self.log(f"Found cached MD file: {md_filepath_cache}. Loading content.")
            try:
                with open(md_filepath_cache, "r", encoding="utf-8") as f_md_cache:
                    cached_md = f_md_cache.read()
                if cached_md and cached_md.strip():
                    final_md_content = cached_md
                    md_available = True
            except Exception as e_read_cache:
                self.log_error(f"Failed to read cached MD {md_filepath_cache}: {e_read_cache}. Will attempt fresh conversion.")
        if not final_md_content and self.mathpix_app_id and self.mathpix_app_key:
            api_md_content = self._convert_pdf_to_mathpix_md(pdf_path)
            if api_md_content:
                final_md_content = api_md_content
                md_available = True
        elif not final_md_content:
             self.log_warning(f"Mathpix conversion skipped or failed for {original_filename}. No MD content from API.")
        raw_text_pages_list: List[str] = []
        pdf_title, pdf_author, pdf_pub_date = None, None, None
        pdf_metadata_extracted = False
        full_raw_text_content = ""
        try:
            pdf_document = fitz.open(pdf_path)
            meta = pdf_document.metadata
            pdf_title = meta.get('title')
            pdf_author = meta.get('author')
            date_str = meta.get('creationDate') or meta.get('modDate') 
            if date_str and date_str.startswith("D:"):
                pdf_pub_date = date_str[2:10]
            else:
                pdf_pub_date = date_str
            pdf_metadata_extracted = bool(pdf_title or pdf_author or pdf_pub_date)
            self.log(f"Extracted PDF metadata: Title='{pdf_title}', Author='{pdf_author}', Date='{pdf_pub_date}'")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text = page.get_text("text")
                raw_text_pages_list.append(text)
                full_raw_text_content += text + "\n"
            pdf_document.close()
            self.log(f"Extracted {len(raw_text_pages_list)} raw text pages from PDF {original_filename}.")
        except Exception as e_fitz:
            self.log_error(f"Error extracting metadata/raw_text via PyMuPDF from {original_filename} ({pdf_path}): {e_fitz}")
            if not final_md_content:
                self.log_error(f"FATAL: No MD content AND PyMuPDF text extraction failed for {original_filename}. Cannot process.")
                return None
        text_for_bib_extraction = final_md_content if final_md_content else full_raw_text_content
        extracted_bibliography = {}
        if text_for_bib_extraction and text_for_bib_extraction.strip():
            try:
                extracted_bibliography = self.text_processor.extract_bibliography(text_for_bib_extraction)
                if extracted_bibliography:
                    self.log(f"Extracted {len(extracted_bibliography)} bibliography items for {original_filename}.")
            except Exception as e_bib:
                 self.log_error(f"Error extracting bibliography for {original_filename}: {e_bib}")
        else:
            self.log_warning(f"No text available for bibliography extraction from {original_filename}.")
        return ScannedPDF(
            document_id=doc_id,
            file_path=pdf_path,
            raw_text_pages=raw_text_pages_list,
            processed_content=final_md_content,
            title=pdf_title or original_filename,
            author=pdf_author, 
            publication_date=pdf_pub_date,
            bibliography=extracted_bibliography,
            metadata_extracted=pdf_metadata_extracted,
            processed_content_available=md_available )
