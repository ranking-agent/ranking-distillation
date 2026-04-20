import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import joblib
import re
from datetime import datetime

from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import torch
from transformers import AutoTokenizer, AutoModel


def sanitize_filename(name):
    clean = re.sub(r'[:\\/]', '_', name)
    clean = clean.replace(' ', '_').lower()
    return re.sub(r'_+', '_', clean).strip('_')


def get_embeddings(sentences, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", batch_size=32):
    """
    Generates embeddings for query sentences using a biomedical BERT model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []
    print(f"Generating embeddings using {model_name} on {device}...")
    
    # Ensure input is numpy/list
    sentence_list = sentences.tolist() if hasattr(sentences, 'tolist') else list(sentences)
    
    for i in tqdm(range(0, len(sentence_list), batch_size)):
        batch_texts = sentence_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token (index 0)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
    return np.vstack(all_embeddings)


def run_lodo_cv(df, X, target_col, run_dir, job_id_col='source_file'):
    """
    Performs LODO CV and saves models and predictions to a unified run directory.
    """
    unique_jobs = df[job_id_col].unique()
    
    print(f"\nStarting LODO CV for target: {target_col}")
    print(f"Saving all artifacts to: {run_dir}")
    
    for test_job in tqdm(unique_jobs, desc="Processing Folds"):
        train_idx = df[df[job_id_col] != test_job].index
        test_idx = df[df[job_id_col] == test_job].index
        
        y_train = df.loc[train_idx, target_col].values
        y_test = df.loc[test_idx, target_col].values
        
        if len(np.unique(y_train)) < 2:
            print(f"Skipping {test_job}: Training set only has one class.")
            continue
            
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        spw = neg_count / pos_count if pos_count > 0 else 1

        clf = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric='logloss', random_state=42
        )
        
        clf.fit(X[train_idx], y_train)
        
        # Save fold model
        safe_name = sanitize_filename(test_job.replace('.parquet', ''))
        model_path = run_dir / f"model_fold_{safe_name}.json"
        clf.save_model(str(model_path))
        
        # Generate predictions
        probs = clf.predict_proba(X[test_idx])[:, 1]
        
        # Save raw predictions alongside the original text for the Streamlit App
        test_df = df.loc[test_idx].copy()
        test_df['y_true'] = y_test      # Standardized column name for the app
        test_df['y_prob'] = probs       # Standardized column name for the app
        
        pred_path = run_dir / f"preds_fold_{safe_name}.parquet"
        test_df.to_parquet(pred_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--target", choices=["is_tier_1", "is_top_tier"], default="is_top_tier")
    args = parser.parse_args()

    # Paths
    RESULTS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/gandalf_responses_predicates_with_inverse/ranking_outputs_flash_3p1_10k")
    MODELS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/models/v3_no_test_assets")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    TEST_ASSET_CURIES = ["CHEBI_27881", "CHEBI_10033"]
    
    # 1. Load Data
    parquet_files = [f for f in RESULTS_DIR.glob("*.parquet") if "-v10-" in f.name and not any([tc in f.name for tc in TEST_ASSET_CURIES])]
    if not parquet_files:
        print("No matching files found."); return

    dfs = [pd.read_parquet(f).assign(source_file=f.name) for f in parquet_files]
    full_df = pd.concat(dfs, ignore_index=True).dropna(subset=['tier', 'query_sentence']).reset_index(drop=True)
    full_df['tier'] = full_df['tier'].astype(int)
    full_df['is_tier_1'] = (full_df['tier'] == 1).astype(int)
    full_df['is_top_tier'] = (full_df['tier'] <= 2).astype(int)

    # 2. Embeddings
    embed_cache_path = MODELS_DIR / "embeddings_v3_cache.joblib"
    if embed_cache_path.exists():
        X = joblib.load(embed_cache_path)
    else:
        X = get_embeddings(full_df['query_sentence'])
        joblib.dump(X, embed_cache_path)

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODELS_DIR / f"run_{timestamp}_{args.target}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        # 3. Training & LODO
        run_lodo_cv(full_df, X, args.target, run_dir)
        
        # Final Model fit
        final_model = XGBClassifier(n_estimators=300, max_depth=6)
        final_model.fit(X, full_df[args.target])
        final_model.save_model(str(run_dir / "final_model.json"))
        
        print(f"Pipeline complete. Run saved to: {run_dir}")

if __name__ == "__main__":
    main()