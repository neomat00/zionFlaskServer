import openai
import logging
import sys
import time
import pdb

from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_embedding_cost(num_tokens):
    num_1k_token_chunks = (num_tokens + 999) // 1000
    return num_1k_token_chunks * 0.0004

def get_pinecone_id_for_file_chunk(bot_name, filename, chunk_index):
    # pdb.set_trace()
    return str(bot_name+"-!"+filename+"-!"+str(chunk_index))

def get_embedding(text, engine):
    return openai.Engine(id=engine).embeddings(input=[text])["data"][0]["embedding"]

def get_embeddings(text_array, engine):
    # pdb.set_trace()
    # Parameters for exponential backoff
    max_retries = 5 # Maximum number of retries
    base_delay = 1 # Base delay in seconds
    factor = 2 # Factor to multiply the delay by after each retry
    while True:
        try:
            return openai.Engine(id=engine).embeddings(input=text_array)["data"]
        except Exception as e:
            if max_retries > 0:
                logging.info(f"Request failed. Retrying in {base_delay} seconds.")
                time.sleep(base_delay)
                max_retries -= 1
                base_delay *= factor
            else:
                raise e