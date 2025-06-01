"""
Pydantic models for representing mathematical text structures like concepts,
references, and scanned PDF documents.
"""
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import uuid4

class Reference(BaseModel):
    """Represents a reference made within a text to another concept or external work."""
    raw_text: str = Field(description="The exact text of the reference (e.g., 'Theorem 3.1', '[Smith2023]', 'eq. (5)')")
    resolved_concept_id: Optional[str] = Field(None, description="The unique ID of the concept this reference points to, if resolved internally.")
    target_document_hint: Optional[str] = Field(None, description="Any hint about the target document (e.g., '[Smith2023]', 'our previous work').")
    target_label_hint: Optional[str] = Field(None, description="Any hint about the target label (e.g., 'Theorem 3.1', 'Section 2').")
    is_resolved: bool = False # Indicates if resolved_concept_id points to a valid internal concept
    external_bibliographic_entry: Optional[str] = Field(None, description="The full bibliographic entry if resolved against a bibliography (e.g., from scanned_pdf.bibliography).")


class ExtractedConcept(BaseModel):
    """Represents a unit of mathematical information extracted from a PDF."""
    concept_id: str = Field(default_factory=lambda: f"concept_{uuid4().hex}", description="Unique identifier for this concept.")
    document_id: str = Field(description="Unique identifier for the source document.")
    concept_type: str = Field(description='e.g., "Definition", "Theorem", "Equation", "Idea", "Proof", "Lemma"')
    label: Optional[str] = Field(None, description="The formal label if any (e.g., 'Theorem 3.1', 'Definition 2', '(5.1)').")
    content: str = Field(description="The actual text/LaTeX/Markdown of the concept.")
    source_file_path: str
    page_number: Optional[int] = Field(None, description="Page number in the source PDF where the concept was found. Currently needs better tracking.") # Changed to Optional[int]
    references_made: List[Reference] = Field(default_factory=list, description="List of references made within this concept's content.")
    llm_model_used: Optional[str] = None
    confidence_score: Optional[float] = None # From specialist agent
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for this concept's content.", exclude=True) # exclude=True means it won't be part of model_dump by default

    def __repr__(self):
        label_part = f" ({self.label})" if self.label else ""
        return f"<{self.concept_type}{label_part}: {self.content[:50]}... (Doc: {self.document_id}, ID: {self.concept_id})>"

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Prepares metadata for ChromaDB storage. Ensures values are Chroma-compatible."""
        return {
            "concept_id": self.concept_id,
            "document_id": self.document_id,
            "concept_type": self.concept_type,
            "label": self.label or "N/A", # Chroma typically wants consistent field presence
            "source_file_path": self.source_file_path,
            "page_number": str(self.page_number) if self.page_number is not None else "N/A", # Chroma prefers strings or numbers
            "references_count": len(self.references_made),
        }

    def get_full_id(self) -> str:
        """Returns a potentially more identifiable ID including the label for logging/debugging."""
        return f"{self.document_id}_{self.concept_type}_{self.label or self.concept_id[-6:]}"

class ScannedPDF(BaseModel):
    """Represents a PDF document that has been processed."""
    document_id: str = Field(default_factory=lambda: f"doc_{uuid4().hex}", description="Unique identifier for this document.")
    file_path: str # This will store the path of the PDF, which can be user-provided
    raw_text_pages: List[str] = Field(default_factory=list)
    processed_content: Optional[str] = Field(None, description="The processed content from the PDF, typically Markdown from Mathpix.")
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    bibliography: Dict[str, str] = Field(default_factory=dict, description="Extracted bibliography (e.g., {'[1]': 'Smith, J. ...', '[Smith23]': 'Smith, J. ...'})")
    metadata_extracted: bool = False
    processed_content_available: bool = False


    def __repr__(self):
        return f"<ScannedPDF: {self.title or os.path.basename(self.file_path)} (ID: {self.document_id})>"