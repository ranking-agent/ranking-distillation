import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import joblib

# Machine Learning
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve
from xgboost import XGBClassifier

# NLP / Embeddings
import torch
from transformers import AutoTokenizer, AutoModel

# Visualization
import matplotlib.pyplot as plt

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
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_texts = sentences[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
    return np.vstack(all_embeddings)

def plot_lodo_performance(lodo_results, target_name, save_dir=None):
    """
    Plots ROC, PR curves, and absolute volume subplots.
    """
    # 2x2 grid to show absolute counts underneath performance curves
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex='col')
    ((ax_roc, ax_pr), (ax_vol_roc, ax_vol_pr)) = axs
    
    lines = []
    labels = []

    for _, row in lodo_results.iterrows():
        y_true = row['y_true']
        y_prob = row['y_prob']
        job_label = row['test_job']
        
        # 1. ROC Curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
        line, = ax_roc.plot(fpr, tpr, label=f"{job_label}")
        lines.append(line)
        labels.append(f"{job_label} (AUC: {row['auc']:.2f} | AP: {row['ap']:.2f})")
        
        # 2. Volume vs FPR (ROC alignment)
        # For each FPR point, how many total positive predictions are there?
        # Total Preds = FPR * Negatives + TPR * Positives
        n_pos = sum(y_true == 1)
        n_neg = sum(y_true == 0)
        total_predicted_roc = (fpr * n_neg) + (tpr * n_pos)
        ax_vol_roc.plot(fpr, total_predicted_roc)

        # 3. PR Curve
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(recall, precision)
        
        # 4. Volume vs Recall (PR alignment)
        # Precision = TP / (TP + FP) -> (TP + FP) = TP / Precision
        # TP = Recall * n_pos
        # Volume = (Recall * n_pos) / Precision
        # Note: handle division by zero at the start of the curve
        vol_pr = np.divide((recall * n_pos), precision, out=np.zeros_like(recall), where=precision!=0)
        ax_vol_pr.plot(recall, vol_pr)
    
    # Formatting
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax_roc.set_ylabel('True Positive Rate (Recall)')
    ax_roc.set_title(f'ROC Curves: {target_name}')
    
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Precision-Recall Curves: {target_name}')
    
    ax_vol_roc.set_xlabel('False Positive Rate')
    ax_vol_roc.set_ylabel('Total Selected Paths (TP + FP)')
    ax_vol_roc.set_title('Volume vs. False Alarm Rate')
    
    ax_vol_pr.set_xlabel('Recall')
    ax_vol_pr.set_ylabel('Total Selected Paths (TP + FP)')
    ax_vol_pr.set_title('Volume vs. Recall')

    for ax in axs.flatten():
        ax.grid(alpha=0.3)

    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
               ncol=1, fontsize='x-small', frameon=True)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"performance_volume_{target_name.lower().replace(' ', '_')}.png"
        plt.savefig(save_path / filename, bbox_inches='tight')
        print(f"Plot saved to: {save_path / filename}")
    
    plt.show()

def run_lodo_cv(df, X, target_col, model_save_dir, job_id_col='source_file'):
    """
    Performs LODO CV and saves weights for each fold.
    """
    unique_jobs = df[job_id_col].unique()
    results = []
    
    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting LODO CV for target: {target_col}")
    
    for test_job in unique_jobs:
        train_idx = df[df[job_id_col] != test_job].index
        test_idx = df[df[job_id_col] == test_job].index
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = df.loc[train_idx, target_col].values, df.loc[test_idx, target_col].values
        
        pos_count = sum(y_train == 1)
        neg_count = sum(y_train == 0)
        spw = neg_count / pos_count if pos_count > 0 else 1

        clf = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric='logloss', random_state=42
        )
        
        clf.fit(X_train, y_train)
        
        # Save fold model
        safe_name = test_job.replace('.parquet', '').replace(' ', '_')
        model_path = model_save_dir / f"model_{target_col}_fold_{safe_name}.json"
        clf.save_model(str(model_path))
        
        probs = clf.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, probs)
            ap = average_precision_score(y_test, probs)
        except ValueError:
            auc, ap = np.nan, np.nan
            
        results.append({
            'test_job': test_job, 'auc': auc, 'ap': ap,
            'y_true': y_test, 'y_prob': probs, 'model_path': model_path
        })
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run training or just evaluation")
    args = parser.parse_args()

    RESULTS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/pathfinder_test_sets/training_queries/ranking_outputs")
    FIGURES_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/figures/surrogate_modeling")
    MODELS_DIR = Path("/mnt/nas0_data1/projects/translator/users/npersson/models/sapbert-xgboost")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    parquet_files = [f for f in RESULTS_DIR.glob("*.parquet") if "-v9-" in f.name]
    if not parquet_files:
        print("No matching files found."); return

    dfs = [pd.read_parquet(f).assign(source_file=f.name) for f in parquet_files]
    full_df = pd.concat(dfs, ignore_index=True).dropna(subset=['tier', 'query_sentence'])
    full_df['tier'] = full_df['tier'].astype(int)
    full_df['is_tier_1'] = (full_df['tier'] == 1).astype(int)
    full_df['is_top_tier'] = (full_df['tier'] <= 2).astype(int)

    # 2. Embeddings (Always needed or could be cached to disk)
    # Check if embeddings are already saved to save time during 'eval' mode
    embed_cache_path = MODELS_DIR / "embeddings_v9_cache.joblib"
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
        plot_lodo_performance(lodo_t1, "Tier 1 vs Rest", save_dir=FIGURES_DIR)
        plot_lodo_performance(lodo_top, "Tier 1 & 2 vs Rest", save_dir=FIGURES_DIR)

    elif args.mode == "eval":
        # 4. Evaluation Only Mode
        # Re-run the logic but loading existing models from MODELS_DIR
        unique_jobs = full_df['source_file'].unique()
        for target in ['is_tier_1', 'is_top_tier']:
            eval_results = []
            for test_job in unique_jobs:
                safe_name = test_job.replace('.parquet', '').replace(' ', '_')
                model_path = MODELS_DIR / f"model_{target}_fold_{safe_name}.json"
                
                if not model_path.exists():
                    print(f"Skipping {test_job}, model not found at {model_path}")
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
                plot_lodo_performance(pd.DataFrame(eval_results), f"EVAL: {target}", save_dir=FIGURES_DIR)

if __name__ == "__main__":
    main()