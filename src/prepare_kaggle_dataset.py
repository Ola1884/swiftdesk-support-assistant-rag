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

def prepare_kaggle_dataset(
        input_filename: str = "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
        output_csv: str = "data/support_conversations.csv",
        output_json: str = "data/test_subset.json",
        test_size: int = 500,
        language_filter: str | None = None,
        ):
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "raw" / input_filename
    output_path = project_root / output_csv
    test_output_path = project_root / output_json

    # check if input files exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Available files in data/raw/: {list((project_root / 'data' / 'raw').glob('*.csv'))}"
        )

    # load dataset
    print(f"Loading dataset from: {input_path.name}")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    print(f"Columns found: {list(df.columns)}")

    # detect colums
    cols = {c.lower().strip(): c for c in df.columns}
    body_cols = next((cols[k] for k in ['body','description','text'] if k in cols),None)
    answer_cols = next((cols[k] for k in ['answer','answers','response'] if k in cols),None)
    lang_cols = next((cols[k] for k in ['language','lang'] if k in cols),None)

    if not all([body_cols,answer_cols,lang_cols]):
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    # filter to English only
    en_df = df[df[lang_cols].str.lower() == "en"].copy()

    en_df['customer_issue'] = en_df[body_cols].astype(str).str.strip()
    en_df['reference_reply'] = en_df[answer_cols].astype(str).str.strip()

    # clean entries
    mask = (en_df['customer_issue'].str.len() > 20) & (en_df['reference_reply'].str.len() > 20)
    final_df = en_df.loc[mask,['customer_issue','reference_reply']].reset_index(drop=True)

    # save output
    output_path.parent.mkdir(parents=True,exist_ok=True)
    final_df.to_csv(output_path,index=False)

    # test subset
    test_records = final_df.head(min(test_size,len(final_df))).to_dict('records')
    with open(test_output_path,'w',encoding='utf-8') as f:
        json.dump(test_records,f,indent=2,ensure_ascii=False)

    return final_df


if __name__ == "__main__":
    prepare_kaggle_dataset()