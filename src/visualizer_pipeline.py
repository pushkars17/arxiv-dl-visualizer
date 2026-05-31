import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Direct internal module routing
from data_processor import load_analytics_data
from dl_classifier import train_dl_layer
from insights_exporter import generate_text_report

os.makedirs("../logs", exist_ok=True)
sns.set_theme(style="whitegrid")

def execute_complete_pipeline():
    csv_path = "../data/arXiv_scientific dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing resource file! Place data at: {csv_path}")
        
    df = load_analytics_data(csv_path)
    
    # -------------------------------------------------------------
    # NETWORK STRUCTURE CALCULATION
    # -------------------------------------------------------------
    print("🕸️  [Analysis] Generating Complex Graph Theory structures...")
    G = nx.Graph()
    for authors_list in df['parsed_authors'].head(20000):
        if len(authors_list) > 1:
            for i in range(len(authors_list)):
                for j in range(i + 1, len(authors_list)):
                    u, v = str(authors_list[i]).strip(), str(authors_list[j]).strip()
                    G.add_edge(u, v, weight=G[u][v]['weight'] + 1 if G.has_edge(u, v) else 1)

    print("🧮 [Analysis] Calculating PageRank metrics...")
    pagerank = nx.pagerank(G, alpha=0.85)
    degree_centrality = nx.degree_centrality(G)
    
    # -------------------------------------------------------------
    # NLP & DEEP LEARNING MODEL EXECUTION
    # -------------------------------------------------------------
    print("🧮 [NLP] Scaling documents to text vectors...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['category_code'])
    num_classes = len(le.classes_)
    
    # Run deep learning functions
    loss_history, confidences = train_dl_layer(tfidf_matrix, y_encoded, num_classes)
    
    # Dump static report ledger
    generate_text_report(df, pagerank, loss_history[-1], output_dir="../logs")

    # -------------------------------------------------------------
    # 7 PREMIUM GRAPHICS PLOTTING ENGINE
    # -------------------------------------------------------------
    print("🎨 [Visualizer] Rendering 7 advanced analysis plots...")

    # Graph 1: Network Topology Log-Scale
    plt.figure(figsize=(10, 5))
    plt.hist(list(degree_centrality.values()), bins=50, color='darkcyan', log=True, edgecolor='black', alpha=0.8)
    plt.title("Graph 1: Academic Collaboration Network Topology Density (Log Scale)", fontweight='bold')
    plt.savefig("../logs/1_network_density_plot.png", dpi=300, bbox_inches='tight'); plt.close()

    # Graph 2: PageRank Core Hubs
    top_authors = pd.DataFrame({'Author': list(pagerank.keys()), 'PageRank': list(pagerank.values())}).sort_values(by='PageRank', ascending=False).head(15)
    plt.figure(figsize=(11, 5))
    sns.barplot(x='PageRank', y='Author', data=top_authors, palette='viridis', edgecolor='black')
    plt.title("Graph 2: Top 15 Dominant Academic Hubs via PageRank Scoring", fontweight='bold')
    plt.savefig("../logs/2_top_authors_hub.png", dpi=300, bbox_inches='tight'); plt.close()

    # Graph 3: Temporal Trends
    top_cats = df['category_code'].value_counts().nlargest(4).index
    trend_data = df[df['category_code'].isin(top_cats)].groupby(['publish_year', 'category_code']).size().unstack(fill_value=0)
    plt.figure(figsize=(11, 5))
    trend_data.plot(kind='line', marker='s', linewidth=2, ax=plt.gca(), cmap='tab10')
    plt.title("Graph 3: Volumetric Evolution of Core Research Domains (1993 - 2025)", fontweight='bold')
    plt.savefig("../logs/3_temporal_trends.png", dpi=300, bbox_inches='tight'); plt.close()

    # Graph 4: Text Vocabulary Weights
    mean_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    terms_df = pd.DataFrame({'Token': vectorizer.get_feature_names_out(), 'Mean_TFIDF': mean_weights}).sort_values(by='Mean_TFIDF', ascending=False).head(25)
    plt.figure(figsize=(11, 5))
    sns.barplot(x='Mean_TFIDF', y='Token', data=terms_df, palette='magma', edgecolor='black')
    plt.title("Graph 4: Top 25 Core Vocabulary Term Loading Weights via TF-IDF", fontweight='bold')
    plt.savefig("../logs/4_text_vocabulary_load.png", dpi=300, bbox_inches='tight'); plt.close()

    # Graph 5: Collaboration Proximity Heatmap
    top_nodes = list(top_authors['Author'].head(10))
    adj_matrix = nx.to_pandas_adjacency(G, nodelist=top_nodes, weight='weight')
    plt.figure(figsize=(9, 7))
    sns.heatmap(adj_matrix, annot=True, fmt=".0f", cmap='Blues', square=True, linewidths=.5, linecolor='gray')
    plt.title("Graph 5: Co-Authorship Proximity Adjacency Interaction Heatmap", fontweight='bold')
    plt.savefig("../logs/5_collaboration_heatmap.png", dpi=300, bbox_inches='tight'); plt.close()

    # Graph 6: NEW - Neural Network Loss Convergence
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='crimson', marker='o', linewidth=2)
    plt.title("Graph 6: PyTorch Deep Learning Model Loss Convergence Curve", fontweight='bold')
    plt.xlabel("Training Epochs Iteration Scale"); plt.ylabel("Cross Entropy Loss Value")
    plt.grid(True, linestyle='--')
    plt.savefig("../logs/6_dl_loss_convergence.png", dpi=300, bbox_inches='tight'); plt.close()

    # Graph 7: NEW - DL Prediction Confidence Density Profile
    plt.figure(figsize=(10, 5))
    sns.kdeplot(confidences, color='forestgreen', fill=True, alpha=0.4, linewidth=2)
    plt.title("Graph 7: Neural Network Softmax Output Target Confidence Profile Density", fontweight='bold')
    plt.xlabel("Probability Prediction Confidence Max Score Value"); plt.ylabel("Density Distribution")
    plt.savefig("../logs/7_dl_confidence_density.png", dpi=300, bbox_inches='tight'); plt.close()

    print("\n" + "👑"*30)
    print("DEEP LEARNING & SEVEN-STAGE GRAPH VISUALIZER PIPELINE RECONSTRUCTED!")
    print("📁 Logs folder holds all 7 advanced high-level premium analytics visualizations plots.")
    print("👑"*30 + "\n")

if __name__ == "__main__":
    execute_complete_pipeline()