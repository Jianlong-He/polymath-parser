"""
Agent dedicated to resolving references found within mathematical concepts,
distinguishing between internal (to the same document or known concepts)
and external (bibliographic) references.
"""
import logging 
from typing import List, Dict, Optional

from .agent import Agent
from .math_text import Reference, ExtractedConcept
from .frontier_agent import MathFrontierAgent

# Import from config
from config import LOG_LEVEL


class ReferencingAgent(Agent):
    name = "Referencing Agent"
    color = Agent.CYAN

    def __init__(self, frontier_agent: MathFrontierAgent): # Removed log_level from __init__ to use config
        log_level_setting = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        super().__init__(log_level=log_level_setting)
        self.frontier = frontier_agent
        self.log("Initialized.")

    def resolve_single_reference(self, reference: Reference, current_doc_id: str, bibliography: Dict[str, str]) -> bool:
        if reference.is_resolved:
            return True
        self.log(f"Processing reference: '{reference.raw_text}' (Label Hint: {reference.target_label_hint}, Doc Hint: {reference.target_document_hint}) in Doc ID: {current_doc_id}")
        resolved_internally = False
        if reference.target_label_hint:
            target_concept = self.frontier.find_concept_by_label(reference.target_label_hint, doc_id=current_doc_id)
            if target_concept:
                reference.resolved_concept_id = target_concept.concept_id
                reference.is_resolved = True
                reference.external_bibliographic_entry = None # Clear this if resolved internally
                self.log(f"Successfully resolved (INTERNAL): '{reference.raw_text}' to concept '{target_concept.get_full_id()}'")
                resolved_internally = True
        
        if not resolved_internally:
            reference.is_resolved = False # Ensure it's False if internal resolution failed or wasn't attempted for label
            reference.resolved_concept_id = None

            bib_key_to_check = reference.target_document_hint or reference.raw_text 
            matched_bib_entry = bibliography.get(bib_key_to_check)
            
            if not matched_bib_entry: # Try a broader check if specific hint didn't match
                for key, entry in bibliography.items():
                    # Check if the key (e.g., "[1]", "[Smith23]") is contained within the raw_text of the reference
                    if key in reference.raw_text:
                        matched_bib_entry = entry
                        break # Found a match
            
            if matched_bib_entry:
                reference.external_bibliographic_entry = matched_bib_entry
                # is_resolved remains False as this is an external, not internal, resolution
                self.log(f"Marked as EXTERNAL (found in bibliography): '{reference.raw_text}' -> '{matched_bib_entry[:60]}...'")
            else:
                # If it has a document hint (like [Smith2023]) or if it has no label hint (making it likely a citation)
                # and it's not in the bibliography, it's likely an unresolved external reference.
                if reference.target_document_hint or not reference.target_label_hint:
                    reference.external_bibliographic_entry = f"External/Unresolved: {reference.raw_text}"
                    self.log_warning(f"Reference '{reference.raw_text}' treated as EXTERNAL (not in bib, has doc_hint or no label_hint).")
                else:
                    # This case means it had a target_label_hint, wasn't found internally, and wasn't found in bib.
                    self.log_warning(f"Could not resolve reference (INTERNAL or AMBIGUOUS): '{reference.raw_text}' with label_hint '{reference.target_label_hint}' in doc '{current_doc_id}'. Not found in bibliography.")
            
        return resolved_internally # Return true only if resolved to an internal concept

    def resolve_references_in_concept_list(self, concepts: List[ExtractedConcept], bibliography: Dict[str, str]) -> int:
        if not concepts:
            return 0
        current_doc_id = concepts[0].document_id if concepts else None 
        if not current_doc_id:
            self.log_error("Cannot resolve references: no document ID context from concepts list.")
            return 0
        self.log(f"Resolving references for {len(concepts)} concepts from document '{current_doc_id}' using bibliography with {len(bibliography)} entries...")
        total_newly_internally_resolved_this_run = 0
        for concept in concepts:
            if concept.document_id != current_doc_id:
                self.log_warning(f"Concept {concept.concept_id} has a different document ID ({concept.document_id}) than the current context ({current_doc_id}). Skipping its references.")
                continue
            for ref_obj in concept.references_made:
                if self.resolve_single_reference(ref_obj, concept.document_id, bibliography):
                    # resolve_single_reference returns True if it was an internal resolution
                    total_newly_internally_resolved_this_run +=1
        self.log(f"Completed reference resolution pass for doc '{current_doc_id}'. Newly internally resolved in this pass: {total_newly_internally_resolved_this_run}.")
        return total_newly_internally_resolved_this_run
