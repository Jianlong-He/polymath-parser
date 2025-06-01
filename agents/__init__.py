"""
Polymath Parser Agents Package

This package contains the various specialized agents used in the Polymath Parser system.
"""

# Import the base Agent class to make it easily accessible
from .agent import Agent

# Import the main concrete agent classes
from .ensemble_agent import MathEnsembleAgent
from .frontier_agent import MathFrontierAgent
from .keyword_extraction_agent import KeywordExtractionAgent
from .pdf_processing_agent import PDFProcessingAgent
from .planning_agent import AnalysisPlanner
from .referencing_agent import ReferencingAgent
from .specialist_agent import MathSpecialistAgent

__all__ = [
    "Agent",
    "MathEnsembleAgent",
    "MathFrontierAgent",
    "KeywordExtractionAgent",
    "PDFProcessingAgent",
    "AnalysisPlanner",
    "ReferencingAgent",
    "MathSpecialistAgent",
]

