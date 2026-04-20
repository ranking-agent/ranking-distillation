import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import joblib
import re

# Machine Learning
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve
from xgboost import XGBClassifier

# NLP / Embeddings
import torch
from transformers import AutoTokenizer, AutoModel

# Visualization
import matplotlib.pyplot as plt


def sanitize_filename(name):
    """
    Sanitizes strings for safe file naming on Mac/Linux/Windows.
    Removes colons, slashes, and spaces.
    """
    # Replace colons, slashes, and backslashes with underscores
    clean = re.sub(r'[:\\/]', '_', name)
    # Replace spaces with underscores and lowercase
    clean = clean.replace(' ', '_').lower()
    # Remove duplicate underscores
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


def plot_lodo_performance(lodo_results, target_name, save_dir=None):
    """
    Plots ROC, PR curves (with monotonic smoothing), and absolute volume subplots.
    """
    fig, axs = plt.subplots(2, 2, figsize=(16, 24), sharex='col')
    ((ax_roc, ax_pr), (ax_vol_roc, ax_vol_pr)) = axs
    
    lines = []
    labels = []

    for _, row in lodo_results.iterrows():
        y_true = row['y_true']
        y_prob = row['y_prob']
        job_label = row['test_job']
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        line, = ax_roc.plot(fpr, tpr, label=f"{job_label}")
        lines.append(line)
        labels.append(f"{job_label} (AUC: {row['auc']:.2f} | AP: {row['ap']:.2f})")
        
        # 2. Volume vs FPR (ROC alignment)
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        total_predicted_roc = (fpr * n_neg) + (tpr * n_pos)
        ax_vol_roc.plot(fpr, total_predicted_roc)

        # 3. PR Curve with Monotonic Smoothing
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        # Enforce monotonicity: P(r) = max_{r' >= r} P(r')
        # precision_smoothed = np.maximum.accumulate(precision[::-1])[::-1]
        
        ax_pr.plot(recall, precision)
        
        # 4. Volume vs Recall (PR alignment)
        # Avoid division by zero
        vol_pr = np.divide((recall * n_pos), precision, out=np.zeros_like(recall), where=precision > 0)
        ax_vol_pr.plot(recall, vol_pr)
    
    # Formatting
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_roc.set_ylabel('True Positive Rate (Recall)')
    ax_roc.set_title(f'ROC Curves: {target_name}')
    
    ax_pr.set_ylabel('Precision')
    ax_pr.set_ylim([0, 1.05])
    ax_pr.set_title(f'Precision-Recall Curves: {target_name}')
    
    ax_vol_roc.set_xlabel('False Positive Rate')
    ax_vol_roc.set_ylabel('Total Selected Paths (TP + FP)')
    ax_vol_roc.set_title('Volume vs. False Alarm Rate')
    
    ax_vol_pr.set_xlabel('Recall')
    ax_vol_pr.set_ylabel('Total Selected Paths (TP + FP)')
    ax_vol_pr.set_title('Volume vs. Recall')
    # Limit volume x-axis to 2000
    ax_vol_pr.set_xlim([0, 2000])

    for ax in axs.flatten():
        ax.grid(alpha=0.3)

    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                ncol=1, fontsize='x-small', frameon=True)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        safe_target = sanitize_filename(target_name)
        filename = f"performance_volume_{safe_target}.png"
        plt.savefig(save_path / filename, bbox_inches='tight')
        print(f"Main plot saved to: {save_path / filename}")
    
    plt.show()


def plot_precision_recall_vs_volume(lodo_results, target_name, save_dir=None):
    """
    Generates a 4-panel figure:
    - Top Left: Precision vs Absolute Volume
    - Top Right: Precision vs Fraction of Dataset
    - Bottom Left: Recall vs Absolute Volume
    - Bottom Right: Recall vs Fraction of Dataset
    """
    # Create 2x2 grid. Share Y across rows to make comparison easier.
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), sharey='row')
    ((ax_p_vol, ax_p_frac), (ax_r_vol, ax_r_frac)) = axs
    
    lines = []
    labels = []

    for _, row in lodo_results.iterrows():
        y_true = np.array(row['y_true'])
        y_prob = np.array(row['y_prob'])
        job_label = row['test_job']
        n_pos = np.sum(y_true == 1)
        
        # Skip folds with no positive samples (fixes blank plots/warnings)
        if n_pos == 0:
            continue

        # Sort indices by probability descending
        idx = np.argsort(y_prob)[::-1]
        y_true_sorted = y_true[idx]
        
        # Calculate raw Precision and Recall at every step
        volumes = np.arange(1, len(y_true_sorted) + 1)
        fractions = volumes / len(y_true_sorted)
        cum_pos = np.cumsum(y_true_sorted)
        
        precision_at_vol = cum_pos / volumes
        recall_at_vol = cum_pos / n_pos

        # Plot Precision (Top Row)
        line, = ax_p_vol.plot(volumes, precision_at_vol, label=job_label, alpha=0.7)
        ax_p_frac.plot(fractions, precision_at_vol, alpha=0.7)
        
        # Plot Recall (Bottom Row)
        ax_r_vol.plot(volumes, recall_at_vol, alpha=0.7)
        ax_r_frac.plot(fractions, recall_at_vol, alpha=0.7)
        
        lines.append(line)
        labels.append(f"{job_label} (AP: {row['ap']:.3f})")

    # Titles and Labels
    ax_p_vol.set_title(f'Precision vs. Absolute Volume: {target_name}')
    ax_p_vol.set_ylabel('Precision')
    ax_p_vol.set_xlim([0, 2000]) # As requested

    ax_p_frac.set_title(f'Precision vs. Fraction of Dataset: {target_name}')
    ax_p_frac.set_xlim([0, 1.0])

    ax_r_vol.set_title(f'Recall vs. Absolute Volume: {target_name}')
    ax_r_vol.set_ylabel('Recall')
    ax_r_vol.set_xlabel('Total Results Chosen (Count)')
    ax_r_vol.set_xlim([0, 2000]) # As requested

    ax_r_frac.set_title(f'Recall vs. Fraction of Dataset: {target_name}')
    ax_r_frac.set_xlabel('Fraction of Dataset (0.0 - 1.0)')
    ax_r_frac.set_xlim([0, 1.0])

    for ax in axs.flatten():
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend at the bottom
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                ncol=2, fontsize='x-small', frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        safe_target = sanitize_filename(target_name)
        plt.savefig(save_path / f"volume_analysis_{safe_target}.png", bbox_inches='tight')
    
    plt.show()


def run_lodo_cv(df, X, target_col, model_save_dir, job_id_col='source_file'):
    """
    Performs LODO CV and skips problematic folds.
    """
    unique_jobs = df[job_id_col].unique()
    results = []
    
    for test_job in tqdm(unique_jobs, desc=f"LODO: {target_col}"):
        train_idx = df[df[job_id_col] != test_job].index
        test_idx = df[df[job_id_col] == test_job].index
        
        y_train = df.loc[train_idx, target_col].values
        y_test = df.loc[test_idx, target_col].values
        
        # Verify training set has both classes
        if len(np.unique(y_train)) < 2:
            print(f"Skipping fold {test_job}: Training set only has one class.")
            continue
            
        # Class weights to handle imbalance
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        spw = neg_count / pos_count if pos_count > 0 else 1

        clf = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric='logloss', random_state=42
        )
        clf.fit(X[train_idx], y_train)
        
        probs = clf.predict_proba(X[test_idx])[:, 1]
        
        # Metrics - handle the case where test set might have 0 positives
        n_pos_test = np.sum(y_test == 1)
        if n_pos_test > 0:
            auc = roc_auc_score(y_test, probs)
            ap = average_precision_score(y_test, probs)
        else:
            # If no positives, metrics are technically undefined or meaningless
            auc, ap = np.nan, np.nan
            
        results.append({
            'test_job': test_job, 'auc': auc, 'ap': ap,
            'y_true': y_test, 'y_prob': probs
        })
        
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run training or just evaluation")
    args = parser.parse_args()

    # Paths (adjust to your environment)
    RESULTS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/gandalf_responses_predicates_with_inverse/ranking_outputs_flash_3p1_10k")
    FIGURES_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/figures/surrogate_modeling/v3_no_test_assets")
    MODELS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/models/v3_no_test_assets")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    TEST_ASSET_CURIES = [
        "CHEBI_27881",
        "CHEBI_10033",
    ]
    
    # 1. Load Data
    parquet_files = [
        f for f in RESULTS_DIR.glob("*.parquet") if "-v10-" in f.name
        and not(any([tc in f.name for tc in TEST_ASSET_CURIES]))
        ]
    
    print(*parquet_files, sep="\n")
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
        print("Loading cached embeddings...")
        X = joblib.load(embed_cache_path)
    else:
        X = get_embeddings(full_df['query_sentence'])
        joblib.dump(X, embed_cache_path)

    if args.mode == "train":
        # 3. Training & LODO
        lodo_t1 = run_lodo_cv(full_df, X, 'is_tier_1', MODELS_DIR)
        lodo_top = run_lodo_cv(full_df, X, 'is_top_tier', MODELS_DIR)
        
        # Final Model fit
        final_model = XGBClassifier(n_estimators=300, max_depth=6)
        final_model.fit(X, full_df['is_top_tier'])
        final_model.save_model(str(MODELS_DIR / "final_model_top_tier.json"))
        
        # Plotting
        for res, name in [
            (lodo_t1, "Tier 1 vs Rest"),
            (lodo_top, "Tier 1 and 2 vs Rest")
            ]:
            plot_lodo_performance(res, name, save_dir=FIGURES_DIR)
            plot_precision_recall_vs_volume(res, name, save_dir=FIGURES_DIR)

    elif args.mode == "eval":
        # 4. Evaluation Only Mode
        unique_jobs = full_df['source_file'].unique()
        for target in [
            'is_tier_1',
            'is_top_tier'
            ]:
            eval_results = []
            for test_job in unique_jobs:
                safe_name = sanitize_filename(test_job.replace('.parquet', ''))
                model_path = MODELS_DIR / f"model_{target}_fold_{safe_name}.json"
                
                if not model_path.exists():
                    print(f"Skipping {test_job}, model not found.")
                    continue
                
                clf = XGBClassifier()
                clf.load_model(str(model_path))
                
                test_idx = full_df[full_df['source_file'] == test_job].index
                y_test = full_df.loc[test_idx, target].values
                X_test = X[test_idx]
                
                probs = clf.predict_proba(X_test)[:, 1]
                eval_results.append({
                    'test_job': test_job, 
                    'auc': roc_auc_score(y_test, probs),
                    'ap': average_precision_score(y_test, probs),
                    'y_true': y_test, 'y_prob': probs
                })
            
            if eval_results:
                df_res = pd.DataFrame(eval_results)
                plot_name = f"EVAL {target.replace('_', ' ').title()}"
                plot_lodo_performance(df_res, plot_name, save_dir=FIGURES_DIR)
                plot_precision_recall_vs_volume(df_res, plot_name, save_dir=FIGURES_DIR)

if __name__ == "__main__":
    main()