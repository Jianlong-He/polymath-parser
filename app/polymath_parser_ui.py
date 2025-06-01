"""
Gradio-based User Interface for the Polymath Parser Agents framework.
Allows users to upload PDFs, choose an LLM model, specify output, trigger analysis, 
view processed documents, extracted concepts, and real-time logs.
"""
import numpy as np
import logging
import queue
import threading
import time
import gradio as gr
import os
from typing import Optional, List, Dict, Any 

# Import from the 'app' package
from .polymath_parser_framework import PolymathParserFramework
# Import from the 'utils' package
from utils import reformat_ansi_to_html
# Import from 'agents'
from agents import Agent

# Import from config
from config import (
    DEFAULT_OUTPUT_DIR as CONFIG_DEFAULT_OUTPUT_DIR, 
    UI_LLM_MODEL_CHOICES, 
    LOG_DISPLAY_HEIGHT,
    MAX_LOG_LINES_IN_MEMORY,
    LOG_LEVEL 
)

GradioFileData = List[Any]

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord):
        formatted_message = self.format(record)
        html_message = reformat_ansi_to_html(formatted_message)
        self.log_queue.put(html_message)

def format_logs_for_html_display(log_data: List[str]) -> str:
    output_html = '<br>'.join(log_data) 
    return f"""
    <div id="logScrollContent" style="overflow-y: scroll; border: 1px solid #444; background-color: #1e1e1e; color: #d4d4d4; padding: 10px; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; line-height: 1.5; min-height: {LOG_DISPLAY_HEIGHT};">
    {output_html}
    </div>
    <script>
        var scrollDiv = document.getElementById('logScrollContent');
        if (scrollDiv) {{ scrollDiv.scrollTop = scrollDiv.scrollHeight; }}
    </script>
    """

def setup_gradio_logging_queue(log_q: queue.Queue):
    logger = logging.getLogger() 
    if not any(isinstance(h, QueueLogHandler) for h in logger.handlers):
        queue_handler = QueueLogHandler(log_q)
        logger.addHandler(queue_handler)

def format_concepts_to_markdown(concepts_data_list: List[Dict], document_title: str) -> str:
    if not concepts_data_list:
        return f"## No concepts extracted for *{document_title}* or document not selected."
    md_output = f"## Extracted Concepts for *{document_title}* ({len(concepts_data_list)})\n\n"
    md_output += "<hr style='margin-bottom: 20px;'>"
    for i, concept_dict in enumerate(concepts_data_list):
        concept_type = concept_dict.get('concept_type', 'N/A')
        label = concept_dict.get('label')
        label_part = f" **({label})**" if label else ""
        md_output += f"### {i+1}. {concept_type}{label_part}\n"
        page_num = concept_dict.get('page_number', 'N/A')
        md_output += f"*Page: {page_num}*\n\n"
        content = concept_dict.get('content', '')
        md_output += f"```text\n{content}\n```\n\n" 
        references_made = concept_dict.get('references_made', [])
        if references_made:
            md_output += "**References Made:**\n"
            for ref_dict in references_made:
                raw_text = ref_dict.get('raw_text', 'Unknown ref')
                is_resolved = ref_dict.get('is_resolved', False)
                resolved_id = ref_dict.get('resolved_concept_id')
                external_entry = ref_dict.get('external_bibliographic_entry')
                status_emoji = "✅ Internal" if is_resolved and resolved_id else \
                               ("🔗 External/Biblio" if external_entry else "❌ Unresolved")
                md_output += f"* `{raw_text}` - {status_emoji}\n"
                if external_entry and not (is_resolved and resolved_id):
                    md_output += f"  *Biblio:* `{external_entry}`\n"
                elif is_resolved and resolved_id:
                    md_output += f"  *Links to ID:* `{resolved_id}`\n"
            md_output += "\n"
        md_output += "<hr style='margin: 15px 0;'>\n\n"
    return md_output

class MathExplorerApp:
    _framework_instance: Optional[PolymathParserFramework] = None
    @classmethod
    def get_agent_framework(cls) -> PolymathParserFramework:
        if cls._framework_instance is None:
            logging.info("UI: Initializing PolymathParserFramework instance...")
            cls._framework_instance = PolymathParserFramework()
        return cls._framework_instance

    def run_ui(self):
        initial_framework_instance = self.get_agent_framework()
        with gr.Blocks(title="Polymath Parser", fill_width=True, theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as polymath_parser_interface:
            log_data_state = gr.State([]) 
            processed_docs_state = gr.State([]) 
            def convert_processed_docs_to_table_data(docs_list: List[Dict]):
                if not docs_list: return []
                table_rows = []
                for item_dict in docs_list:
                    scanned_doc_info = item_dict.get('scanned_document', {})
                    display_name = item_dict.get('original_pdf_filename') or \
                                   scanned_doc_info.get('title') or \
                                   os.path.basename(scanned_doc_info.get('file_path', 'Unknown File'))
                    num_concepts = len(item_dict.get('concepts', []))
                    resolved_refs = item_dict.get('resolved_references_count', 0)
                    path_identifier = scanned_doc_info.get('file_path', 'N/A')
                    table_rows.append([display_name, num_concepts, resolved_refs, path_identifier])
                return table_rows

            def analysis_worker_target(framework_instance: PolymathParserFramework, pdf_tmp_files: GradioFileData, output_dir: str, llm_model_id_choice: str, log_q: queue.Queue, result_q: queue.Queue[List[Dict]]):
                setup_gradio_logging_queue(log_q) 
                pdf_actual_paths = [tmp_file.name for tmp_file in pdf_tmp_files if hasattr(tmp_file, 'name')]
                logging.info(f"UI Worker: Started for {len(pdf_actual_paths)} files. Output: {output_dir}, LLM: {llm_model_id_choice}")
                if not pdf_actual_paths:
                    logging.error("UI Worker: No valid PDF file paths received.")
                    result_q.put([])
                    log_q.put(reformat_ansi_to_html(f"{Agent.RED}[ERROR]{Agent.RESET} Error: No PDF files were processed.")) # Use Agent colors
                    return
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    updated_docs_data = framework_instance.run_analysis_cycle(
                        pdf_file_paths=pdf_actual_paths, output_directory=output_dir,
                        llm_model_identifier=llm_model_id_choice )
                    result_q.put(updated_docs_data)
                    log_msg = f"--- Analysis Run Complete for {len(pdf_actual_paths)} file(s). Output in {output_dir} ---"
                    logging.info(f"UI Worker: Finished. Results queued. {log_msg}") 
                except Exception as e_worker:
                    logging.error(f"UI Worker: Error during analysis: {e_worker}", exc_info=True)
                    result_q.put([]) 

            def run_analysis_threaded_and_stream_updates(pdf_files_from_ui: Optional[GradioFileData], output_dir_str: str, llm_model_choice: str, current_logs_state: List[str]):
                if not pdf_files_from_ui:
                    err_msg = f"{Agent.RED}[ERROR]{Agent.RESET} No PDF files uploaded. Please select PDF files to analyze."
                    formatted_err_msg = reformat_ansi_to_html(err_msg)
                    updated_logs = (current_logs_state + [formatted_err_msg])[-MAX_LOG_LINES_IN_MEMORY:]
                    yield updated_logs, format_logs_for_html_display(updated_logs), convert_processed_docs_to_table_data([]), []
                    return
                final_output_dir = output_dir_str.strip() if output_dir_str.strip() else os.path.join(os.getcwd(), CONFIG_DEFAULT_OUTPUT_DIR)
                log_q, result_q = queue.Queue(), queue.Queue[List[Dict]]()
                framework_ref = self.get_agent_framework()
                setup_gradio_logging_queue(log_q)
                worker_thread = threading.Thread(target=analysis_worker_target, args=(framework_ref, pdf_files_from_ui, final_output_dir, llm_model_choice, log_q, result_q), daemon=True)
                worker_thread.start()
                temp_logs_list = list(current_logs_state)
                temp_docs_list: List[Dict] = [] 
                analysis_results_received_flag = False # To ensure we process results at least once after worker finishes
                while worker_thread.is_alive() or not log_q.empty() or (not result_q.empty() and not analysis_results_received_flag):
                    logs_updated_this_batch = False
                    while not log_q.empty():
                        try: temp_logs_list.append(log_q.get_nowait()); logs_updated_this_batch = True
                        except queue.Empty: break
                    if logs_updated_this_batch: temp_logs_list = temp_logs_list[-MAX_LOG_LINES_IN_MEMORY:]
                    
                    if not result_q.empty() and not analysis_results_received_flag: # Process results only once if worker is done
                        try:
                            final_docs_result = result_q.get_nowait()
                            temp_docs_list = final_docs_result if isinstance(final_docs_result, list) else []
                            analysis_results_received_flag = True # Mark results as processed
                            logging.info(f"UI Stream: Received analysis results. {len(temp_docs_list)} items.")
                        except queue.Empty: pass 
                    
                    current_table_view = convert_processed_docs_to_table_data(temp_docs_list)
                    yield temp_logs_list, format_logs_for_html_display(temp_logs_list), current_table_view, temp_docs_list
                    
                    if not worker_thread.is_alive() and log_q.empty() and (result_q.empty() or analysis_results_received_flag):
                        break # Exit if worker done, logs empty, and results processed or result queue also empty
                    time.sleep(0.3)
                
                # Final update to ensure everything is flushed
                while not log_q.empty(): temp_logs_list.append(log_q.get_nowait())
                temp_logs_list = temp_logs_list[-MAX_LOG_LINES_IN_MEMORY:]
                if not analysis_results_received_flag and not result_q.empty(): # One last check for results
                     try: temp_docs_list = result_q.get_nowait() if isinstance(result_q.get_nowait(), list) else []
                     except queue.Empty: pass
                final_table_view = convert_processed_docs_to_table_data(temp_docs_list)
                yield temp_logs_list, format_logs_for_html_display(temp_logs_list), final_table_view, temp_docs_list

            def handle_document_selection(all_docs_data_list: List[Dict], evt: gr.SelectData) -> str:
                if not all_docs_data_list or evt is None or not hasattr(evt, 'index') or evt.index is None or not evt.index:
                    return "## Select a document from the table to view its extracted concepts."
                try:
                    selected_row_index = evt.index[0]
                    if selected_row_index >= len(all_docs_data_list): raise IndexError("Selected row index out of bounds.")
                    selected_item_dict = all_docs_data_list[selected_row_index]
                    doc_title = selected_item_dict.get('original_pdf_filename') or \
                                selected_item_dict.get('scanned_document', {}).get('title') or \
                                os.path.basename(selected_item_dict.get('scanned_document', {}).get('file_path', 'Unknown Title'))
                    logging.info(f"UI: Displaying concepts for selected document: '{doc_title}' at index {selected_row_index}.")
                    concepts_for_selected_doc = selected_item_dict.get('concepts', [])
                    return format_concepts_to_markdown(concepts_for_selected_doc, doc_title)
                except IndexError:
                     logging.error(f"UI Error: Selected index {evt.index[0] if evt.index else 'N/A'} out of bounds for {len(all_docs_data_list)} documents.")
                     return "## Error: Selected document index is out of bounds."
                except Exception as e_sel:
                    logging.error(f"UI Error displaying concepts: {e_sel}", exc_info=True)
                    return f"## Error displaying concepts:\n{str(e_sel)}"

            gr.Markdown('<div style="text-align: center; font-size: 28px; margin-bottom: 5px; color: #2A6098;"><strong>🧭 Polymath Parser</strong></div><div style="text-align: center; font-size: 14px; margin-bottom: 20px; color: #555;">Convert mathematical PDFs to structured JSON with selectable LLM models.</div>')
            with gr.Row():
                with gr.Column(scale=2): pdf_upload_input = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDF(s) for Analysis")
                with gr.Column(scale=1):
                    llm_model_selector_dropdown = gr.Dropdown(choices=UI_LLM_MODEL_CHOICES, label="Select LLM Model", value=UI_LLM_MODEL_CHOICES[0] if UI_LLM_MODEL_CHOICES else None)
                    output_dir_input = gr.Textbox(label="Output Directory for JSON files", placeholder=f"Default: {os.path.join(os.getcwd(), CONFIG_DEFAULT_OUTPUT_DIR)}")
            run_analysis_button = gr.Button("🚀 Start Analysis", variant="primary", size="lg")
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Processed Documents (Current Session/Output Directory)")
                    insights_dataframe = gr.Dataframe(headers=["Document Name", "# Concepts", "# Res Refs", "Path/ID"], datatype=["str", "number", "number", "str"], row_count=(5, "dynamic"), wrap=True, interactive=True)
                    load_from_dir_button = gr.Button("🔄 Load/Refresh Results from Output Directory")
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Extracted Concepts Detail")
                    concept_display_markdown = gr.Markdown(value="## Select a document from the table to view its extracted concepts.", elem_id="concept-display-markdown")
            gr.Markdown("---")
            gr.Markdown("### 🪵 Real-time Logs")
            logs_html_output = gr.HTML(value=format_logs_for_html_display([]), elem_id="logs-html-output")

            def initial_load_ui_elements():
                framework = self.get_agent_framework()
                default_load_dir = os.path.join(os.getcwd(), CONFIG_DEFAULT_OUTPUT_DIR)
                initial_docs_in_memory = framework.load_memory_from_directory(default_load_dir)
                initial_table_content = convert_processed_docs_to_table_data(initial_docs_in_memory)
                welcome_log_msg = "[SYSTEM] Welcome! Upload PDFs, choose settings, and click 'Start Analysis'."
                welcome_log = [reformat_ansi_to_html(welcome_log_msg)]
                initial_logs_html = format_logs_for_html_display(welcome_log)
                logging.info(f"UI loaded. Attempted to load from: {default_load_dir}")
                return welcome_log, initial_logs_html, initial_table_content, initial_docs_in_memory

            polymath_parser_interface.load(initial_load_ui_elements, inputs=[], outputs=[log_data_state, logs_html_output, insights_dataframe, processed_docs_state])
            run_analysis_button.click(run_analysis_threaded_and_stream_updates, inputs=[pdf_upload_input, output_dir_input, llm_model_selector_dropdown, log_data_state], outputs=[log_data_state, logs_html_output, insights_dataframe, processed_docs_state])
            
            def load_results_from_specified_dir(output_dir_path_str: str, current_logs: List[str]):
                load_path = output_dir_path_str.strip() if output_dir_path_str.strip() else os.path.join(os.getcwd(), CONFIG_DEFAULT_OUTPUT_DIR)
                framework = self.get_agent_framework()
                loaded_docs = framework.load_memory_from_directory(load_path)
                table_data = convert_processed_docs_to_table_data(loaded_docs)
                log_msg = f"[SYSTEM] Loaded {len(loaded_docs)} items from directory: {load_path}"
                logging.info(log_msg.replace("[SYSTEM] ", ""))
                updated_logs = (current_logs + [reformat_ansi_to_html(log_msg)])[-MAX_LOG_LINES_IN_MEMORY:]
                return updated_logs, format_logs_for_html_display(updated_logs), table_data, loaded_docs

            load_from_dir_button.click(load_results_from_specified_dir, inputs=[output_dir_input, log_data_state], outputs=[log_data_state, logs_html_output, insights_dataframe, processed_docs_state])
            insights_dataframe.select(handle_document_selection, inputs=[processed_docs_state], outputs=[concept_display_markdown])
        polymath_parser_interface.queue(default_concurrency_limit=10).launch(server_name="0.0.0.0", inbrowser=True, debug=False) 

if __name__ == "__main__":
    app_runner = MathExplorerApp()
    app_runner.run_ui()
