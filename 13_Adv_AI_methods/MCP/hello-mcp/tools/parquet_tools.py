from server import mcp
from utils.file_reader import read_parquet_summary
import logging

logger = logging.getLogger(__name__)

@mcp.tool()
def summarize_parquet_file(filename: str) -> str:
    """
    Summarize a Parquet file with detailed statistics.
    Args:
        filename: Name of the Parquet file in the /data directory (e.g., 'sample.parquet')
    Returns:
        A detailed summary of the Parquet file including:
        - Number of rows and columns
        - Data types of columns
        - Memory usage
        - Compression details
    """
    try:
        summary = read_parquet_summary(filename)
        logger.info(f"Successfully summarized Parquet file: {filename}")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing Parquet file {filename}: {str(e)}")
        raise