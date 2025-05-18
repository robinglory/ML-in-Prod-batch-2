
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Base directory where our data lives
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def read_csv_summary(filename: str) -> str:
    """
    Read a CSV file and return a detailed summary.
    Args:
        filename: Name of the CSV file (e.g. 'sample.csv')
    Returns:
        A string describing the file's contents with detailed statistics.
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: Other errors during file processing
    """
    try:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        df = pd.read_csv(file_path)
        summary = [
            f"CSV file '{filename}' summary:",
            f"- Rows: {len(df)}",
            f"- Columns: {len(df.columns)}",
            "- Column types:",
            *[f"  * {col}: {str(dtype)}" for col, dtype in zip(df.columns, df.dtypes)],
            f"- Memory usage: {df.memory_usage().sum() / (1024*1024):.2f} MB",
            f"- File size: {file_path.stat().st_size / (1024*1024):.2f} MB"
        ]
        return "\n".join(summary)
    except FileNotFoundError as e:
        logger.error(f"File not found: {filename}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV file {filename}: {str(e)}")
        raise

def read_parquet_summary(filename: str) -> str:
    """
    Read a Parquet file and return a detailed summary.
    Args:
        filename: Name of the Parquet file (e.g. 'sample.parquet')
    Returns:
        A string describing the file's contents with detailed statistics.
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: Other errors during file processing
    """
    try:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        df = pd.read_parquet(file_path)
        print(df.shape)
        row, col  = df.shape
        
        if row > 100:
            df = df.head(100)
        summary = [
            f"Parquet file '{filename}' summary:",
            f"- Rows: {len(df)}",
            f"- Columns: {len(df.columns)}",
            "- Column types:",
            *[f"  * {col}: {str(dtype)}" for col, dtype in zip(df.columns, df.dtypes)],
            f"- Memory usage: {df.memory_usage().sum() / (1024*1024):.2f} MB",
            f"- File size: {file_path.stat().st_size / (1024*1024):.2f} MB",
            #f"- Compression: {df.get_engine()}",
        ]
        return "\n".join(summary)
    except FileNotFoundError as e:
        logger.error(f"File not found: {filename}")
        raise
    except Exception as e:
        logger.error(f"Error reading Parquet file {filename}: {str(e)}")
        raise