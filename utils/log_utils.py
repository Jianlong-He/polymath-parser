"""
Utility for converting ANSI color codes in log messages to HTML spans
for display in environments that support HTML (like Gradio's HTML component).
"""

# ANSI Foreground colors
ANSI_RED = '\033[31m'
ANSI_GREEN = '\033[32m'
ANSI_YELLOW = '\033[33m'
ANSI_BLUE = '\033[34m'
ANSI_MAGENTA = '\033[35m'
ANSI_CYAN = '\033[36m'
ANSI_WHITE = '\033[37m'

# ANSI Background colors
ANSI_BG_BLACK = '\033[40m'
ANSI_BG_BLUE = '\033[44m' # Defined but not currently used in MAPPER

# ANSI Reset code
ANSI_RESET = '\033[0m'

# Mapping from ANSI color combinations to HTML color styles
# Used by Agent base class's formatter
ANSI_TO_HTML_COLOR_MAP = {
    f"{ANSI_BG_BLACK}{ANSI_RED}": "#dd0000",      # Red
    f"{ANSI_BG_BLACK}{ANSI_GREEN}": "#00dd00",    # Green
    f"{ANSI_BG_BLACK}{ANSI_YELLOW}": "#dddd00",   # Yellow
    f"{ANSI_BG_BLACK}{ANSI_BLUE}": "#0000ee",     # Blue
    f"{ANSI_BG_BLACK}{ANSI_MAGENTA}": "#aa00dd", # Magenta
    f"{ANSI_BG_BLACK}{ANSI_CYAN}": "#00dddd",     # Cyan
    f"{ANSI_BG_BLACK}{ANSI_WHITE}": "#B0C4DE",   # LightSteelBlue (was 87CEEB SkyBlue, more subtle)
    # Example if BG_BLUE was used by a logger:
    # f"{ANSI_BG_BLUE}{ANSI_WHITE}": "#ff7800",  # Orange on blue background (example)
}


def reformat_ansi_to_html(ansi_message: str) -> str:
    """
    Converts ANSI color codes within a log message string to HTML spans
    for colored display in HTML-supporting interfaces.

    :param ansi_message: The log message string possibly containing ANSI codes.
    :return: The message string with ANSI codes replaced by HTML color spans.
    """
    html_message = ansi_message
    for ansi_code_combo, html_color_value in ANSI_TO_HTML_COLOR_MAP.items():
        # Ensure the replacement creates a valid span tag
        html_message = html_message.replace(ansi_code_combo, f'<span style="color: {html_color_value};">')
    
    # Replace all ANSI reset codes with closing span tags
    # This assumes that a color span was opened before any reset.
    # More robust parsing might be needed for complex/nested ANSI sequences,
    # but for typical colored log lines, this should work.
    html_message = html_message.replace(ANSI_RESET, '</span>')
    
    # Ensure any unclosed spans (e.g., if message ends mid-color without reset) are closed.
    # This is a simple heuristic. A proper parser would track open tags.
    open_spans = html_message.count('<span')
    closed_spans = html_message.count('</span>')
    if open_spans > closed_spans:
        html_message += '</span>' * (open_spans - closed_spans)
        
    return html_message

# Original `reformat` function name kept for compatibility if used elsewhere directly by that name.
# It's good practice to use the more descriptive name internally.
reformat = reformat_ansi_to_html