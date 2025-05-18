from server import mcp
from utils.file_reader import read_csv_summary
import logging

logger = logging.getLogger(__name__)

@mcp.tool()
def summarize_csv_file(filename: str) -> str:
    """
    Summarize a CSV file with detailed statistics.
    Args:
        filename: Name of the CSV file in the /data directory (e.g., 'sample.csv')
    Returns:
        A detailed summary of the CSV file including:
        - Number of rows and columns
        - Data types of columns
        - Memory usage
    """
    try:
        summary = read_csv_summary(filename)
        logger.info(f"Successfully summarized CSV file: {filename}")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing CSV file {filename}: {str(e)}")
        raise