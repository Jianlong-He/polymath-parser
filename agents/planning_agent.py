"""
Agent responsible for coordinating the research workflow.
Now processes a given list of PDF files and saves to a specified output directory.
"""
from typing import List, Dict, Optional
from datetime import datetime
import os
import logging


from .agent import Agent
from .math_text import ScannedPDF, ExtractedConcept
from .pdf_processing_agent import PDFProcessingAgent
from .ensemble_agent import MathEnsembleAgent
from .frontier_agent import MathFrontierAgent
from .referencing_agent import ReferencingAgent

# Import from config
from config import LOG_LEVEL

class AnalysisPlanner(Agent):
    name = "Analysis Planner"
    color = Agent.GREEN

    def __init__(self, frontier_agent: MathFrontierAgent, 
                 llm_model_identifier: str):
        log_level_setting = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        super().__init__(log_level=log_level_setting)
        self.log("Initializing...")
        self.scanner = PDFProcessingAgent() 
        self.frontier = frontier_agent
        self.referencing_agent = ReferencingAgent(self.frontier)
        self.extractor = MathEnsembleAgent(
            self.frontier, 
            self.referencing_agent, 
            llm_model_identifier=llm_model_identifier
        )
        self.log("Ready.")

    def run_analysis_on_document(self, scanned_pdf: ScannedPDF) -> Dict:
        self.log(f"Analyzing: {scanned_pdf.file_path} (Title: {scanned_pdf.title or 'N/A'})")
        if not (scanned_pdf.processed_content or scanned_pdf.raw_text_pages):
            self.log_error(f"PDF {scanned_pdf.title or scanned_pdf.file_path} has no processable content. Skipping analysis.")
            return {
                "scanned_document": scanned_pdf.model_dump(mode='json'),
                "concepts": [],
                "processed_timestamp": datetime.now().isoformat(),
                "resolved_references_count": 0,
                "status": "failed_no_content"
            }
        extracted_concepts_models, resolved_count = self.extractor.analyze(scanned_pdf)
        processed_data = {
            "original_pdf_filename": os.path.basename(scanned_pdf.file_path), 
            "scanned_document": scanned_pdf.model_dump(mode='json'),
            "concepts": [c.model_dump(mode='json') for c in extracted_concepts_models],
            "processed_timestamp": datetime.now().isoformat(),
            "resolved_references_count": resolved_count,
            "status": "processed_successfully"
        }
        self.log(f"Analysis complete for {scanned_pdf.title or scanned_pdf.file_path}. Extracted {len(extracted_concepts_models)} concepts. Resolved {resolved_count} refs.")
        return processed_data

    def process_pdf_files(self, pdf_file_paths: List[str]) -> List[Dict]: # Renamed from process_pdf_files_batch for consistency
        self.log(f"Received {len(pdf_file_paths)} PDF files for processing.")
        if not pdf_file_paths:
            self.log("No PDF files provided. Processing complete.")
            return []
        newly_processed_data_items: List[Dict] = []
        for pdf_path in pdf_file_paths:
            self.log(f"Starting processing for PDF: {pdf_path}")
            scanned_pdf_model = self.scanner.process_pdf(pdf_path) 
            if scanned_pdf_model:
                processed_item_dict = self.run_analysis_on_document(scanned_pdf_model)
                newly_processed_data_items.append(processed_item_dict)
                concepts_to_store_models = []
                if processed_item_dict.get("status") == "processed_successfully":
                    try:
                        concepts_to_store_models = [ExtractedConcept(**c_dict) for c_dict in processed_item_dict['concepts']]
                    except Exception as e:
                        self.log_error(f"Error reconstructing ExtractedConcept models for {pdf_path}: {e}")
                if concepts_to_store_models:
                    self.log(f"Storing {len(concepts_to_store_models)} concepts from {os.path.basename(pdf_path)} into vector store...")
                    self.frontier.store_concepts(concepts_to_store_models)
            else:
                self.log_error(f"Failed to scan/process PDF: {pdf_path}. Skipping this file.")
        self.log(f"Planner finished processing batch of {len(pdf_file_paths)} PDFs. Generated {len(newly_processed_data_items)} new data sets.")
        return newly_processed_data_items
