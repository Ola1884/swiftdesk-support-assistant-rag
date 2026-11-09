"""
Prepare Kaggle Multilingual Customer Support Tickets dataset.

Converts the raw Kaggle CSV into the project format:
- customer_issue: Combined subject + body
- reference_reply: Agent's answer
"""
import pandas as pd
import os
import json
from pathlib import Path
import logging
import numpy as np


# set up logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
KNOWLEDGE_BASE_CSV = PROJECT_ROOT / "data" / "support_conversations.csv"
TEST_SUBSET_JSON = PROJECT_ROOT / "data" / "test_subset.json"

def prepare_kaggle_dataset():

    # 1. Load raw data
    df = pd.read_csv(RAW_CSV,low_memory=False)

    # 2. Filter language to English only
    if 'language' in df.columns:
        df = df[df['language'].astype(str).str.lower() == 'en']

    # 3. Clean and format (subject + body -> customer issue),(answer -> reference reply)
    df["customer_issue"] = df["subject"].fillna('') + "." + df["body"].fillna('')
    df["customer_issue"] = df["customer_issue"].str.replace(r'\s+','',regex=True).str.strip()
    df["reference_reply"] = df["answer"].fillna('').str.strip()

    # 4. Drop empty or duplicate columns
    df = df[(df["customer_issue"] != '') & (df["reference_reply"] != '')]
    df = df.drop_duplicates(subset=["customer_issue"]).reset_index(drop=True)

    # 5. Shuffle dataset
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)

    # 6. Create subset from the dataset 
    # 800 rows for chroma db
    # 15 row for test subset
    project_subset = df.head(n=815).copy()

    # Split data -> train & test
    test_subset = project_subset.head(n=15).copy()
    train_subset = project_subset.tail(n=800).copy()

    # 7. Save knowledge base for chroma ingestion
    train_subset[['customer_issue','reference_reply']].to_csv(KNOWLEDGE_BASE_CSV,index=False)
    logger.info(f"Saved {len(train_subset)} rows to {KNOWLEDGE_BASE_CSV} (For ChromaDB)")

    test_subset[['customer_issue','reference_reply']].to_json(TEST_SUBSET_JSON,orient='records',indent=4)
    logger.info(f"Saved {len(test_subset)} rows to {TEST_SUBSET_JSON} (For ROUGE-L Evaluation)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_kaggle_dataset()