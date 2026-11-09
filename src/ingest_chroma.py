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

    # 1. Convert DataFrame to LangChain Documents
    logger.info("Converting DataFrame to LangChain Documents...")
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row['customer_issue'],
            metadata={"reference_reply":row['reference_reply']}
        )
        documents.append(doc)

    # 2. Create embeddings model
    logger.info("Creating embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 3. Ingest documents into Chroma
    logger.info("Ingesting documents into Chroma...")
    batch_size = 100
    vectorstore= None
    for i in tqdm(range(0, len(documents), batch_size),desc="Ingesting Documents"):
        batch = documents[i:i + batch_size]
        #  Create DB on the first batch, then add to it
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_PATH),
                collection_name=COLLECTION_NAME
            )
        else:
            vectorstore.add_documents(batch)

    logger.info(f"Successfully ingested {len(documents)} documents into Chroma at {CHROMA_PATH}.")

if __name__ == "__main__":
    ingest_data()
    



    

    



