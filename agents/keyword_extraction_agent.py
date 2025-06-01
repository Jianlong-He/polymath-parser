"""
Agent that extracts potential keywords, concept labels (e.g., Theorem 3.1),
and citation markers (e.g., [Smith2023], [12]) from text.
"""
import re
from typing import List, Tuple

from .agent import Agent

class KeywordExtractionAgent(Agent):
    """
    Extracts potential keywords, concept labels, and citation markers from text.
    """
    name = "Keyword Extractor"
    color = Agent.MAGENTA

    CONCEPT_PATTERN = re.compile(
        r'\b(Theorem|Lemma|Proposition|Corollary|Definition|Equation|Section|Chapter|Remark|Figure|Table|Appendix|Algorithm|Axiom|Conjecture|Example)\s+([A-Z]?\d+(\.\d+)*([a-z]?)?)\b',
        re.IGNORECASE
    )
    CITATION_PATTERN = re.compile(
        r'(\[[\w\d\s.,&;-]+?\]' # Matches [AuthorYear], [1], [A. B. et al., 2023]
        r'|\([\w\s\.,&;-]+?\d{4}[\w\s\.,&;-]*?\)' # Matches (Author, Year) or (Author et al. Year)
        r'|\{\w+\})' # Matches {BibTeXKey} - less common in final text
    ) 

    def __init__(self):
        super().__init__()
        self.log("Ready for keyword/label extraction.")

    def extract_keywords_and_references(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extracts potential concept labels and citation markers.
        :param text: The text chunk to analyze.
        :return: A tuple containing (list_of_unique_labels, list_of_unique_citations).
        """
        identified_labels = []
        for match in self.CONCEPT_PATTERN.finditer(text):
            identified_labels.append(f"{match.group(1)} {match.group(2)}")

        identified_citations = self.CITATION_PATTERN.findall(text)
        
        if identified_labels or identified_citations:
            self.log(f"Found {len(set(identified_labels))} unique potential labels and {len(set(identified_citations))} unique potential citations.")
        
        return list(set(identified_labels)), list(set(identified_citations))
