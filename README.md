# Multimodal Academic Intelligence: Hybrid Deep Learning & Graph Theory Network Pipeline

An enterprise-grade, high-performance data engineering and analytics pipeline designed to extract structural insights from over 136,000 scientific research publications (186 MB corpus). This system bypasses standard visual dashboards to deliver an automated backend architecture that fuses **Complex Graph Theory Networks** with **Deep Learning Text Embedding Convergence** to map out academic evolution, collaboration networks, and semantic patterns.

## 🏗️ System Architecture & Data Flow

The backend operates as a modular engineering pipeline, segregating structural data cleaning, matrix tokenization, network topological mining, and neural network tensor backpropagation.

[Raw Data] ──> [Data Processor] ──┬──> [Network Graph Engine] ──> (Topological Features & 5 Graph Plots)
│
└──> [PyTorch Neural Net] ───> (Tensor Classification & 2 DL Plots)


---

## 📂 Repository Structure

```text
arxiv-dl-visualizer/
│
├── data/
│   └── arXiv_scientific dataset.csv  # 186MB raw corpus (Strictly Gitignored)
│
├── logs/                             # Automated Telemetry & Visual Analytics Output
│   ├── 1_network_density_plot.png     # Graph 1: Scale-free network distribution
│   ├── 2_top_authors_hub.png          # Graph 2: PageRank authority mapping
│   ├── 3_temporal_trends.png          # Graph 3: Historical domain volume shifts
│   ├── 4_text_vocabulary_load.png     # Graph 4: TF-IDF corpus token weights
│   ├── 5_collaboration_heatmap.png    # Graph 5: Co-authorship adjacency matrix
│   ├── 6_dl_loss_convergence.png      # Graph 6: PyTorch training loss curve
│   ├── 7_dl_confidence_density.png    # Graph 7: Softmax prediction distribution profile
│   └── executive_summary_report.txt  # Comprehensive mathematical metrics ledger
│
├── src/                              # Core Source Engine
│   ├── __init__.py                   # Package initialization marker
│   ├── data_processor.py             # High-speed data parsing & text token cleaning
│   ├── dl_classifier.py              # PyTorch Multi-Layer Perceptron architecture
│   ├── insights_exporter.py          # Automated management matrix compiler
│   └── visualizer_pipeline.py        # Central Orchestration Engine
│
├── .gitignore                        # Prevents cluttering of datasets, charts, and cache
├── README.md                         # Documentation
└── requirements.txt                  # Strict environment tracking layout
🚀 Key Engineering Core Features
1. Complex Graph Topology Mining (NetworkX)
Mathematical Nodes & Edges: Constructs massive structural co-authorship networks scanning thousands of relationships dynamically mapped from string representations of list structures.

PageRank & Centrality Tracking: Deploys Google’s native PageRank routing logic alongside Degree Centrality algorithms to calculate exact domain influencers and structural network scale bridges.

2. Deep Learning NLP Engine (PyTorch)
Multi-Layer Perceptron Layers: Feeds sparse TF-IDF text representations into a fully connected custom Neural Network utilizing nn.ReLU and nn.Dropout(0.3) layers to prevent over-fitting.

Tensor Backpropagation: Manages dynamic loss calculations utilizing an optimized Adam engine running over CrossEntropyLoss matrix profiles.

3. Automated Telemetry Design Patterns
Eliminates heavy frontend dependencies or visual framework overhead.

Generates an executive data ledger (executive_summary_report.txt) recording cross-sectional domain statistics, system scales, and neural network configurations directly into automated pipeline logs.

Spits out 7 premium production-ready analytics plots charting everything from scale-free networks to neural network confidence curves.

🛠️ Installation & Execution Guidelines
1. Clone the Repository & Setup Data
Bash
git clone [https://github.com/your-username/arxiv-dl-visualizer.git](https://github.com/pushkars17/arxiv-dl-visualizer.git)
cd arxiv-dl-visualizer
⚠️ Important Step: Place your downloaded arXiv_scientific dataset.csv directly inside the data/ directory before triggering execution.

2. Environment Setup & Dependency Resolution
Windows paths or global registry launchers can sometimes disrupt direct binary calls. Bypass global locks using Python's standard modular engine path commands:

Bash
python -m pip install -r requirements.txt
3. Execute the Central Pipeline
Navigate to the source directory and execute the main orchestration script:

Bash
cd src
python visualizer_pipeline.py
📊 Automated Visual Assets Output
Upon successful execution, the machine learning and graph engine generates the following 7 diagnostic assets inside the logs/ directory:

1_network_density_plot.png: Visualizes scale-free academic node frequency densities across a log scale distribution matrix.

2_top_authors_hub.png: Maps out the top 15 dominant researchers based on Google PageRank influence metrics.

3_temporal_trends.png: Outlines historical line graphs tracing the evolution and shift of core machine learning fields from 1993 to 2025.

4_text_vocabulary_load.png: Displays a horizontal bar metric graphing the token weight loading across all text feature abstractions.

5_collaboration_heatmap.png: Illustrates structural cross-sectional collaboration density charts using proximity matrix fields.

6_dl_loss_convergence.png: Plots the descending cross-entropy error metric mapping PyTorch optimization layers over iterations.

7_dl_confidence_density.png: Uses Kernel Density Estimation (KDE) to plot the distribution of final target layer softmax prediction confidences.

📜 Developer Profile
Developer: Pushkar Sharma

Specialization: Artificial Intelligence, Data Analytics & Backend Machine Learning Systems

Focus: High-Performance Computing, Structural Data Pipeline Engineering & Applied Mathematical Modeling