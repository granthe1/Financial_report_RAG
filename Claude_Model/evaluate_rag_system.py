import pandas as pd
import json
import os
import random
from datetime import datetime
from rag_with_claude import initialize_models, create_vector_store, create_qa_chain, load_and_process_pdfs
# 新增导入
try:
    from sentence_transformers import SentenceTransformer, util
    _st_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    _st_model = None

QA_FILE = "Q&A.csv"
PDF_FOLDER = "10kFiles"
SAMPLE_SIZE = 150
CONFIG_FILE = "rag_config.json"

# Candidate system prompts (all in English)
CANDIDATE_PROMPTS = [
    "You are a financial analyst. Answer strictly based on the provided 10-K documents. If the answer is not in the documents, say you don't know.",
    "You are an expert in financial report analysis. Use only the information from the given 10-K files to answer. Do not make up information.",
    "You are a business consultant. Answer the user's question using only the context from the uploaded 10-K reports. If unsure, say so.",
    "You are a professional analyst. Provide detailed, accurate answers based only on the 10-K context. If the answer is not found, say 'Insufficient information.'"
]

DEFAULT_PARAMS = {
    "chunk_size": 1500,
    "chunk_overlap": 300,
    "num_chunks": 8,
    "temperature": 0.05,
    "max_tokens": 4000
}

def load_qa_csv(file_path: str) -> pd.DataFrame:
    # Try multiple encodings for compatibility
    encodings = ['utf-8-sig', 'gbk', 'latin1']
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            df = df[df["Question"].notnull() & df["Answer"].notnull()]
            return df
        except Exception as e:
            last_error = e
            continue
    raise last_error

def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    groups = df.groupby(["Company", "Category"])
    samples = []
    per_group = max(1, n // len(groups))
    for _, group in groups:
        take = min(per_group, len(group))
        samples.append(group.sample(n=take, random_state=42))
    result = pd.concat(samples)
    if len(result) < n:
        left = df.drop(result.index)
        if len(left) > 0:
            extra = left.sample(n=n-len(result), random_state=42)
            result = pd.concat([result, extra])
    return result.sample(frac=1, random_state=42).reset_index(drop=True)

def evaluate_answer(pred: str, gold: str) -> dict:
    if not pred or not gold:
        return {"correct": 0, "relevance": 0, "completeness": 0}
    # 语义相似度判定
    if _st_model is not None:
        try:
            emb_pred = _st_model.encode(pred, convert_to_tensor=True)
            emb_gold = _st_model.encode(gold, convert_to_tensor=True)
            sim = float(util.pytorch_cos_sim(emb_pred, emb_gold).item())
            relevance = 1 if sim >= 0.75 else 0
        except Exception:
            relevance = 0
    else:
        # 回退到关键词重叠
        pred_set = set(pred.lower().split())
        gold_set = set(gold.lower().split())
        overlap = len(pred_set & gold_set)
        relevance = 1 if overlap > 2 else 0
    completeness = 1 if len(pred) > 30 else 0
    correct = 1 if relevance and completeness else 0
    return {"correct": correct, "relevance": relevance, "completeness": completeness}

def run_eval_for_prompt(prompt: str, sample_df: pd.DataFrame, params: dict) -> dict:
    print(f"\nEvaluating with prompt: {prompt[:60]}...")
    llm, embeddings = initialize_models()
    chunks = load_and_process_pdfs()
    vector_store, retriever = create_vector_store(chunks, embeddings)
    qa_chain = create_qa_chain(llm, retriever)
    results = []
    for idx, row in sample_df.iterrows():
        q = row["Question"]
        gold = row["Answer"]
        print(f"[{idx+1}/{len(sample_df)}] Q: {q[:60]}...")
        try:
            response = qa_chain.invoke({"query": q, "system_prompt": prompt})
            pred = response["result"]
            sources = response.get("source_documents", [])
        except Exception as e:
            pred = f"Error: {e}"
            sources = []
        eval_score = evaluate_answer(pred, gold)
        results.append({
            "question": q,
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": eval_score["correct"],
            "relevance": eval_score["relevance"],
            "completeness": eval_score["completeness"],
            "sources_used": len(sources)
        })
    correct = sum(r["correct"] for r in results)
    relevance = sum(r["relevance"] for r in results)
    completeness = sum(r["completeness"] for r in results)
    avg_sources = sum(r["sources_used"] for r in results) / len(results)
    summary = {
        "prompt": prompt,
        "total": len(results),
        "correct": correct,
        "relevance": relevance,
        "completeness": completeness,
        "avg_sources": avg_sources,
        "results": results
    }
    return summary

def main():
    print("Loading Q&A data...")
    df = load_qa_csv(QA_FILE)
    print(f"Total Q&A pairs: {len(df)}")
    sample_df = stratified_sample(df, SAMPLE_SIZE)
    print(f"Sampled {len(sample_df)} questions for evaluation.")
    best_score = -1
    best_summary = None
    for prompt in CANDIDATE_PROMPTS:
        summary = run_eval_for_prompt(prompt, sample_df, DEFAULT_PARAMS)
        print(f"Prompt: {prompt[:60]}... | Correct: {summary['correct']} | Relevance: {summary['relevance']}")
        if summary["correct"] > best_score:
            best_score = summary["correct"]
            best_summary = summary
    print("\n===== Best Prompt Evaluation Summary =====")
    print(f"Prompt: {best_summary['prompt']}")
    print(f"Total: {best_summary['total']}")
    print(f"Correct: {best_summary['correct']} ({best_summary['correct']/best_summary['total']:.2%})")
    print(f"Relevance: {best_summary['relevance']} ({best_summary['relevance']/best_summary['total']:.2%})")
    print(f"Completeness: {best_summary['completeness']} ({best_summary['completeness']/best_summary['total']:.2%})")
    print(f"Avg Sources Used: {best_summary['avg_sources']:.2f}")
    # Save best config
    config = DEFAULT_PARAMS.copy()
    config["system_prompt"] = best_summary["prompt"]
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Best config saved to {CONFIG_FILE}")
    # Save detailed results
    out_file = f"rag_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(best_summary["results"], f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {out_file}")

if __name__ == "__main__":
    main() 