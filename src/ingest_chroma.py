import pandas as pd
from pathlib import Path
from tqdm import tqdm 
import os
from dotenv import load_dotenv
import sys
import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# import settings
try:
    from src.settings import settings
except ImportError:
    settings = None

# logging settings
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

#======= Configuration =======
PROJECT_ROOT =  Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "support_conversations.csv"

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "support_tickets"

def ingest_data():
    # check if dataset path exists
    if not DATA_PATH.exists():
        logger.error(f"Dataset path {DATA_PATH} does not exist.")
        sys.exit(1)
    
    # check if api key is set
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY is not set.")
        sys.exit(1)

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        logger.error("Dataset is empty.")
        sys.exit(1)

    # create embeddings
    logger.info("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector = embeddings.embed_documents(ids = ["issue","reply"],documents = [df["customer_issue"].tolist(),df["reference_reply"].tolist()])


    

    



