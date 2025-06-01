"""
Main orchestrator for the Polymath Parser application.
Manages the analysis planner, data persistence, 
and interfaces with the vector database.
"""
import os
import sys
import logging
import json
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
import numpy as np
import chromadb

# Import agents using the 'agents' package
from agents import AnalysisPlanner, MathFrontierAgent, Agent

# Import from config
from config import (
    DEFAULT_OUTPUT_DIR as CONFIG_DEFAULT_OUTPUT_DIR, 
    DB_PATH as CONFIG_DB_PATH,
    DB_COLLECTION_NAME as CONFIG_DB_COLLECTION_NAME,
    LOG_LEVEL
)

def init_logging():
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        log_level_setting = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        root_logger.setLevel(log_level_setting)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", 
            datefmt="%H:%M:%S" )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        noisy_loggers = ['httpx', 'httpcore', 'chromadb.telemetry', 'sentence_transformers.SentenceTransformer']
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger('chromadb').setLevel(logging.WARNING)

class PolymathParserFramework(Agent): # Inherits from agents.Agent
    DB_PATH = CONFIG_DB_PATH
    COLLECTION_NAME = CONFIG_DB_COLLECTION_NAME

    _last_plot_data: Optional[Tuple[List[Dict[str, Any]], np.ndarray, List[str]]] = None
    _last_db_count: int = -1

    def __init__(self):
        log_level_setting = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        super().__init__(log_level=log_level_setting) 
        self.name = "Polymath Parser Framework" 
        self.color = Agent.BLUE 
        init_logging() 
        load_dotenv()
        self.log("Framework initializing...")
        self.default_output_directory = os.path.abspath(CONFIG_DEFAULT_OUTPUT_DIR)
        os.makedirs(self.default_output_directory, exist_ok=True)
        self.log(f"Default JSON output directory: {self.default_output_directory}")
        self.frontier = MathFrontierAgent(collection_name=self.COLLECTION_NAME)
        self.memory: List[Dict] = [] 
        self.log("Framework ready. Memory is initially empty or loaded on demand by UI.")

    def load_memory_from_directory(self, directory_path: str) -> List[Dict]:
        processed_data_list: List[Dict] = []
        abs_dir_path = os.path.abspath(directory_path)
        self.log(f"Reading memory from {abs_dir_path}...")
        if not os.path.isdir(abs_dir_path):
            self.log_error(f"Memory directory {abs_dir_path} not found or is not a directory. Cannot load.")
            self.memory = []
            return []
        try:
            for filename in sorted(os.listdir(abs_dir_path)):
                if filename.lower().endswith('.json'):
                    file_path = os.path.join(abs_dir_path, filename)
                    try:
                        with open(file_path, "r", encoding='utf-8') as mem_file:
                            data_item = json.load(mem_file)
                            if 'scanned_document' in data_item and 'concepts' in data_item:
                                processed_data_list.append(data_item)
                            else:
                                self.log_warning(f"Memory file {filename} in {abs_dir_path} is missing essential keys. Skipping.")
                    except json.JSONDecodeError as e_json:
                        self.log_error(f"Error decoding JSON from {file_path}: {e_json}")
                    except Exception as e_read:
                        self.log_error(f"Error reading file {file_path}: {e_read}")
        except Exception as e_list:
            self.log_error(f"Error listing memory directory {abs_dir_path}: {e_list}")
        self.memory = processed_data_list
        self.log(f"Loaded {len(self.memory)} items from {abs_dir_path}.")
        return self.memory

    def _save_single_processed_item_to_disk(self, processed_item_dict: Dict, output_directory: str) -> None:
        try:
            base_filename_for_output = "processed_output" 
            original_pdf_filename = processed_item_dict.get('original_pdf_filename')
            if original_pdf_filename:
                 base_filename_for_output = os.path.splitext(original_pdf_filename)[0]
            elif processed_item_dict.get('scanned_document', {}).get('title'):
                base_filename_for_output = os.path.splitext(processed_item_dict['scanned_document']['title'])[0]
            safe_filename = "".join(c if c.isalnum() or c in (' ', '.', '_', '-') else '_' for c in base_filename_for_output).rstrip()
            json_filename = safe_filename + ".json"
            os.makedirs(output_directory, exist_ok=True)
            output_path = os.path.join(output_directory, json_filename)
            with open(output_path, "w", encoding='utf-8') as out_file:
                json.dump(processed_item_dict, out_file, indent=2)
            self.log(f"Saved processed data for '{base_filename_for_output}' to {output_path}")
        except Exception as e_write:
            failed_doc_title = processed_item_dict.get('scanned_document', {}).get('title', 'Unknown document')
            self.log_error(f"Error writing data for {failed_doc_title} to disk: {e_write}")

    def run_analysis_cycle(self, pdf_file_paths: List[str], 
                           output_directory: str, 
                           llm_model_identifier: str) -> List[Dict]:
        self.log(f"Starting analysis cycle for {len(pdf_file_paths)} PDFs. Output to: {output_directory}. LLM Identifier: {llm_model_identifier}")
        analysis_planner = AnalysisPlanner( 
            frontier_agent=self.frontier, 
            llm_model_identifier=llm_model_identifier )
        newly_processed_data_items = analysis_planner.process_pdf_files(pdf_file_paths) 
        self.log(f"Analysis Planner returned {len(newly_processed_data_items)} new data sets.")
        if newly_processed_data_items:
            for item_dict in newly_processed_data_items:
                self._save_single_processed_item_to_disk(item_dict, output_directory) 
            self.memory = newly_processed_data_items 
            self.log(f"Framework memory updated with {len(self.memory)} items from the current run (output: {output_directory}).")
        else:
            self.memory = [] 
            self.log(f"No new items processed. Framework memory is now empty (reflecting run to {output_directory}).")
        self.log(f"Analysis cycle completed for output directory: {output_directory}.")
        return self.memory

    @classmethod
    def get_visualization_data(cls, max_datapoints: int = 1000) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str]]:
        logger = logging.getLogger(f"{cls.__name__}.get_visualization_data")
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.error("Scikit-learn (sklearn) is not installed. Cannot generate t-SNE plot data. Please install it (`pip install scikit-learn`).")
            return ([], np.array([]), [])
        default_return = ([], np.array([]), [])
        try:
            client = chromadb.PersistentClient(path=cls.DB_PATH) 
            collection = client.get_or_create_collection(cls.COLLECTION_NAME) 
            current_db_count = collection.count()
            if current_db_count == cls._last_db_count and cls._last_plot_data is not None:
                logger.info(f"Using cached plot data (DB count: {current_db_count}).")
                return cls._last_plot_data
            logger.info(f"DB count changed (was {cls._last_db_count}, now {current_db_count}). Recalculating plot data...")
            cls._last_db_count = current_db_count
            if current_db_count == 0:
                 cls._last_plot_data = default_return
                 logger.info("No data in DB for plotting.")
                 return cls._last_plot_data
            limit = min(max_datapoints, current_db_count) 
            retrieved_data = collection.get(include=['embeddings', 'metadatas'], limit=limit)
            if not retrieved_data or not retrieved_data.get('embeddings') or not retrieved_data['embeddings']:
                cls._last_plot_data = default_return
                logger.warning("No embeddings found in retrieved data from DB.")
                return cls._last_plot_data
            embeddings_array = np.array(retrieved_data['embeddings'])
            metadatas_list = retrieved_data['metadatas']
            if embeddings_array.shape[0] < 4:
                 if embeddings_array.shape[1] >= 3: vectors_3d = embeddings_array[:, :3]
                 elif embeddings_array.shape[1] == 2: vectors_3d = np.hstack((embeddings_array, np.zeros((embeddings_array.shape[0], 1))))
                 elif embeddings_array.shape[1] == 1: vectors_3d = np.hstack((embeddings_array, np.zeros((embeddings_array.shape[0], 2))))
                 else: vectors_3d = np.zeros((embeddings_array.shape[0], 3))
                 logger.info(f"Using direct embedding dimensions (padded to 3D if needed) for plot due to few points ({embeddings_array.shape[0]}).")
            else:
                perplexity_value = min(30, embeddings_array.shape[0] - 1)
                tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity_value, n_iter=300, init='pca', learning_rate='auto')
                vectors_3d = tsne.fit_transform(embeddings_array)
                logger.info(f"t-SNE transformation complete for {vectors_3d.shape[0]} points.")
            unique_concept_types = sorted(list(set(m.get('concept_type', 'Unknown') for m in metadatas_list)))
            color_map = {ctype: f'hsl({i * (360 // (len(unique_concept_types) or 1))}, 70%, 60%)' for i, ctype in enumerate(unique_concept_types)}
            plot_colors = [color_map.get(m.get('concept_type', 'Unknown'), '#808080') for m in metadatas_list]
            cls._last_plot_data = (metadatas_list, vectors_3d, plot_colors)
            return cls._last_plot_data
        except Exception as e_plot:
            logger.error(f"Framework: Error getting plot data: {e_plot}", exc_info=True)
            cls._last_plot_data = default_return
            return default_return
