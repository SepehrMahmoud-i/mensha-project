# --- File and Directory Paths ---
RAW_DATA_PATH = "./data/all_combined_data.json"
CHUNKED_DATA_PATH = "./data/doctors_final_enriched_chunks.jsonl" 
DB_PATH = "./db" 
COLLECTION_NAME = "doctors_collection"

# --- Model Names ---
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL = "gpt-4o"

# --- Processing Parameters ---
BATCH_SIZE = 100
BIO_WORD_LIMIT = 154
MAX_WORDS_PER_CHUNK = 450
INITIAL_SEARCH_RESULTS = 20 # Fetch more results for better re-ranking
TOP_N_FOR_CONTEXT = 5      # Use top 5 doctors for the LLM context
FINAL_RECOMMENDATIONS = 3  # Ask the LLM to recommend 3

# --- API Info ---
# IMPORTANT: Paste your secret API Key here
GAPGPT_API_KEY = "sk-hrgk5T1hRGqYl7De6B6nETsjff0mx8JjUpCGVfXoiIDU2w0h" 
GAPGPT_API_BASE_URL = "https://api.gapgpt.app/v1"