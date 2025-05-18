from server import mcp
import tools.csv_tools 
import tools.parquet_tools 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    try:
        logger.info("Starting mix-server...")
        mcp.run(transport='stdio')
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)


if __name__ == "__main__":


    #tools.parquet_tools.summarize_parquet_file("/Users/tharhtet/Documents/github/mcp_testing/mix_server/data/green_tripdata_2022-02.parquet")
    main()
