"""
Agent that combines insights from various specialist agents to extract and link
mathematical concepts from a document.
"""
from typing import List, Tuple, Dict, Any, Optional
import logging

from .agent import Agent
from .specialist_agent import MathSpecialistAgent
from .frontier_agent import MathFrontierAgent
from .keyword_extraction_agent import KeywordExtractionAgent
from .referencing_agent import ReferencingAgent
from .math_text import ExtractedConcept, Reference, ScannedPDF

# Import MathTextProcessor from the 'processing' package
from processing import MathTextProcessor

class MathEnsembleAgent(Agent):
    name = "Math Ensemble"
    color = Agent.YELLOW

    def __init__(self, frontier_agent: MathFrontierAgent, 
                 referencing_agent: ReferencingAgent,
                 llm_model_identifier: str):
        super().__init__()
        self.log(f"Initializing with LLM Model Identifier: {llm_model_identifier}")
        
        self.specialist = MathSpecialistAgent(llm_model_identifier=llm_model_identifier)
        self.frontier = frontier_agent 
        self.keyword_agent = KeywordExtractionAgent()
        self.text_processor = MathTextProcessor() # From processing package
        self.referencing_agent = referencing_agent
        self.log("Ready.")

    def analyze(self, scanned_pdf: ScannedPDF) -> Tuple[List[ExtractedConcept], int]:
        self.log(f"Running analysis for: {scanned_pdf.file_path}")
        all_concepts: List[ExtractedConcept] = []
        doc_id = scanned_pdf.document_id

        text_to_process = scanned_pdf.processed_content or "\n\n".join(scanned_pdf.raw_text_pages)
        if not text_to_process.strip():
            self.log_warning(f"No content (processed or raw text) to process for {scanned_pdf.file_path}. Skipping analysis.")
            return [], 0
            
        scrubbed_text = self.text_processor.scrub_text(text_to_process)
        chunks = self.text_processor.chunk_text(scrubbed_text)
        self.log(f"Processing {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            self.log(f"Processing chunk {i+1}/{len(chunks)}...")
            
            similar_concepts = self.frontier.find_similar_concepts(chunk, n_results=10)
            rag_context = "\n---\n".join([f"Related: {c.content[:150]}..." for c in similar_concepts])
            
            specialist_results = self.specialist.extract_structured_info(chunk, rag_context)

            for result in specialist_results:
                try:
                    refs_data = result.get("references", [])
                    parsed_references = []
                    for r_data in refs_data:
                        parsed_references.append(Reference(
                            raw_text=r_data.get("raw_text", "Unknown reference text"),
                            target_document_hint=r_data.get("target_document_hint"),
                            target_label_hint=r_data.get("target_label_hint")
                        ))

                    content_value = result.get("content") or chunk
                    all_concepts.append(ExtractedConcept(
                        document_id=doc_id, 
                        concept_type=result.get("concept_type", "Idea"),
                        label=result.get("label"), 
                        content=content_value,
                        source_file_path=scanned_pdf.file_path, 
                        page_number=0, # TODO: Page number tracking
                        references_made=parsed_references,
                        llm_model_used=f"{self.specialist.ui_model_identifier} (Provider: {self.specialist.llm_provider})",
                        confidence_score=result.get("confidence_score")
                    ))
                except Exception as e:
                    self.log_error(f"Failed to parse specialist result: {result}. Error: {e}")

        self.log(f"Extracted {len(all_concepts)} initial concepts.")

        resolved_count = 0
        if all_concepts:
            self.log("Handing off to ReferencingAgent for reference resolution...")
            resolved_count = self.referencing_agent.resolve_references_in_concept_list(
                all_concepts,
                scanned_pdf.bibliography
            )
            self.log(f"ReferencingAgent resolved {resolved_count} references.")
        else:
            self.log("No concepts extracted, skipping reference resolution.")

        return all_concepts, resolved_count
