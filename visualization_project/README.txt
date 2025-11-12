
GraphCLAR — Homogeneous Dash (CSV)

Run locally:
  python -m venv .venv
  source .venv/bin/activate       # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  python app.py
  Open the printed local URL.

Data:
  Place teammates.csv and teammates_deps.csv in the data/ folder (already copied if available here).
  Edges CSV should contain columns: source, target, type, (optional) weight.
  Nodes CSV should contain: id, name (or path), (optional) module, node_type.

Controls:
  - Choose edge types to include (homogeneous graph, filtered by type)
  - Choose PCA or t-SNE for 2D projection
  - Choose color by: Cluster / Module / Node Type
  - Graph tab shows induced subgraph using embedding coordinates

Plug in your encoder:
  - Replace compute/encoders.py:get_demo_embedding with your GAT/DGI pipeline.
  - Keep the function signature and return a NumPy array Z (N x d).

