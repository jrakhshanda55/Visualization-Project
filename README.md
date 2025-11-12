\section*{Software Architecture Analysis Dashboard (GNNs + Graph Autoencoders)}

An \textbf{interactive dashboard} for exploring and visualizing software architecture using 
\textbf{Graph Neural Networks (GNNs)} and \textbf{Graph Autoencoders (GAEs)}. 
The app allows users to analyze dependencies, view architectural structures, 
and train GNN-based models to learn software module representations.

\subsection*{Features}
\begin{itemize}
    \item Load and explore software dependency datasets
    \item Visualize implemented and intended architectures
    \item View module and dependency type distributions
    \item Train \textbf{GAE} or \textbf{VGAE} models using \textbf{GCN} or \textbf{GAT} encoders
    \item Compare 2D embedding projections using \textbf{UMAP} or \textbf{t-SNE}
\end{itemize}

\subsection*{Requirements and Running the Dashboard}

All dependencies are listed in \texttt{requirements.txt}. Install them using:

\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}

Then run the dashboard with:

\begin{verbatim}
python app.py
\end{verbatim}

After starting, open your browser and visit:

\begin{verbatim}
http://127.0.0.1:8050/
\end{verbatim}

You will see the \textbf{Software Architecture Analysis Dashboard}, where you can:
\begin{itemize}
    \item Select datasets from the sidebar
    \item Explore dependency and module distributions
    \item Visualize implemented and intended architectures
    \item Train GNN models and view learned embeddings
\end{itemize}

\subsection*{Project Structure}
\begin{verbatim}
visualization_project/
│
├── app.py                         # Main dashboard application
├── requirements.txt               # Dependencies
├── compute/
│   ├── data_builder.py            # Data loading and preprocessing
│   ├── gnn_models.py              # GNN and GAE/VGAE models
│   ├── projection.py              # 2D projection (UMAP/t-SNE)
│   └── plot_architecture.py       # Architecture visualization
└── data/
    ├── <dataset_name>.csv         # Node data
    ├── <dataset_name>_deps.csv    # Dependency data
    └── <dataset_name>_relations.csv (optional intended architecture)
\end{verbatim}

\subsection*{Example}
When the app is running, the dashboard displays:
\begin{itemize}
    \item \textbf{Dependency Type Distribution}
    \item \textbf{Module Distribution}
    \item \textbf{Implemented and Intended Architecture Graphs}
    \item \textbf{2D Embedding Projection} of learned representations
\end{itemize}

\subsection*{Summary}
Interactive dashboard for visualizing software architecture using GNN-based autoencoders.
