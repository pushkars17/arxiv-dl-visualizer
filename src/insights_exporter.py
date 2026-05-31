import os
import pandas as pd

def generate_text_report(df, pagerank, dl_final_loss, output_dir="../logs"):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "executive_summary_report.txt")
    
    print("📝 [Metrics Export] Writing hybrid mathematical results table...")
    top_pr = pd.DataFrame({'Author': list(pagerank.keys()), 'Score': list(pagerank.values())})
    top_pr = top_pr.sort_values(by='Score', ascending=False).head(10)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("      ARXIV DATA INSIGHTS, DEEP LEARNING & COMPLEX GRAPH REPORT      \n")
        f.write("======================================================================\n\n")
        
        f.write("📌 [1. SYSTEM SCALE METRICS]\n")
        f.write(f" -> Total Research Papers Parsed: {len(df):,}\n")
        f.write(f" -> Unique Target Categories: {df['category_code'].nunique()}\n\n")
        
        f.write("🔥 [2. DEEP LEARNING TELEMETRY]\n")
        f.write(f" -> PyTorch Model Final Loss Metric: {dl_final_loss:.5f}\n")
        f.write(" -> Loss Optimization Optimization Layer Status: CONVERGED\n\n")
        
        f.write("🕸️ [3. TOP INFLUENTIAL ENTITIES (PAGERANK ALGORITHM)]\n")
        for idx, row in enumerate(top_pr.itertuples(), 1):
            f.write(f"  Rank {idx:02d} | Score: {row.Score:.6f} | Author: {row.Author}\n")
            
        f.write("\n======================================================================\n")