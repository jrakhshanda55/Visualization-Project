from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np, pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ---- compute modules ----
from compute.data_builder import load_nodes_edges, available_dependency_types, build_dataset
from compute.gnn_models import train_gae
from compute.projection import project_2d
from compute.plot_architecture import plot_implemented_architecture, plot_intended_architecture


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Software Architecture Analysis (GNNs + Graph Auto-Encoders)"

# --------------------------
# ---------- Data ----------
# --------------------------
DATA_DIR = Path(r"C:\Users\JABEERAK\Architecture_Recovery\visualization_project\data")

def _find_dataset_pairs(data_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    nodes_map, deps_map = {}, {}
    if not data_dir.exists():
        return {}
    for p in data_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        stem = p.stem
        if stem.endswith("_deps"):
            deps_map[stem[:-5]] = p
        else:
            nodes_map[stem] = p
    return {b: (nodes_map[b], deps_map[b]) for b in nodes_map if b in deps_map}

DATASETS = _find_dataset_pairs(DATA_DIR)
DATASET_OPTIONS = [{"label": k, "value": k} for k in sorted(DATASETS.keys())]
DEFAULT_DATASET = DATASET_OPTIONS[0]["value"] if DATASET_OPTIONS else None

# ---------- Styles ----------
BASE_FONT = {"fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif", "color": "#0c1127"}
LABEL_STYLE = {"fontSize": "20px", "fontWeight": "600", "marginTop": "8px", **BASE_FONT}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0, "left": 0, "bottom": 0,
    "width": "28rem",
    "padding": "1.8rem 1.4rem",
    "backgroundColor": "#f8f9fa",
    "overflowY": "auto",
    "borderRight": "1px solid #ddd",
    "fontSize": "18px",
    "lineHeight": "1.6",
    **BASE_FONT
}
CONTENT_STYLE = {
    "marginLeft": "30rem",
    "padding": "1rem 1.6rem",
    "backgroundColor": "#fff",
    "minHeight": "100vh",
    **BASE_FONT
}

def mini_kpi(title, comp_id):
    return dbc.Col(
        html.Div(
            [
                html.Div(title, className="text-muted",
                         style={"fontSize": "20px", "fontWeight": "500"}),
                html.Div(id=comp_id, className="fw-bold",
                         style={"fontSize": "26px", "color": "#0d47a1"})
            ],
            className="p-2 text-center",
            style={"border": "1px solid #ddd", "borderRadius": "8px",
                   "backgroundColor": "#fff", "minWidth": "110px",
                   "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}
        ),
        width="auto"
    )

# ---------- Sidebar ----------
dataset_controls = dbc.Card(
    [
        dbc.CardHeader(html.H4("Dataset", style={
            "textAlign": "center", "fontWeight": "700", "fontSize": "24px", "color": "#1a237e"})),
        dbc.CardBody([
            dcc.Dropdown(id="dataset", options=DATASET_OPTIONS, value=DEFAULT_DATASET, clearable=False,
                         persistence=True, persistence_type="memory", style={"fontSize": "18px"}),
            html.Br(),
            html.P("Top-N Modules", style=LABEL_STYLE),
            dcc.Slider(id="topn", min=5, max=20, step=1, value=10,
                       marks={i: str(i) for i in [5, 10, 15, 20]},
                       tooltip={"always_visible": False, "placement": "bottom"})
        ])
    ],
    style={"marginBottom": "20px", "border": "1px solid #ddd", "borderRadius": "8px",
           "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}
)

gnn_controls = dbc.Accordion([
    dbc.AccordionItem([
        html.P("Dependency Types", style=LABEL_STYLE),
        dcc.Dropdown(id="dep-types", options=[], value=[], multi=True,
                     placeholder="Select dependency types...", style={"fontSize": "18px"}),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.P("Encoder", style=LABEL_STYLE),
                dbc.RadioItems(id="enc-type",
                               options=[{"label": "GAT", "value": "gat"},
                                        {"label": "GCN", "value": "gcn"}],
                               value="gat", inline=True, style={"fontSize": "18px"})
            ], width=6),
            dbc.Col([
                html.P("Model", style=LABEL_STYLE),
                dbc.RadioItems(id="model-type",
                               options=[{"label": "GAE", "value": "gae"},
                                        {"label": "VGAE", "value": "vgae"}],
                               value="gae", inline=True, style={"fontSize": "18px"})
            ], width=6)
        ]),
        html.Br(),
        html.P("Features", style=LABEL_STYLE),
        dbc.RadioItems(id="feat-mode",
                       options=[{"label": "File location + Code (W2V)", "value": "file_location+code_w2v"},
                                {"label": "File identifier", "value": "simple"}],
                       value="file_location+code_w2v", style={"fontSize": "18px"}),
        html.Br(),
        html.P("Embedding Dim", style=LABEL_STYLE),
        dcc.Slider(id="hidden-dim", min=64, max=256, step=32, value=64,
                   marks={64: "64", 128: "128", 256: "256"}),
        html.Br(),
        html.P("Epochs", style=LABEL_STYLE),
        dcc.Slider(id="epochs", min=30, max=100, step=10, value=30,
                   marks={30: "30", 50: "50", 100: "100"}),
        html.Br(),
        html.P("Projection", style=LABEL_STYLE),
        dbc.RadioItems(id="proj-method",
                       options=[{"label": "UMAP", "value": "umap"},
                                {"label": "t-SNE", "value": "tsne"}],
                       value="tsne", inline=True, style={"fontSize": "18px"}),
        dbc.Row(dbc.Col(
            dbc.Button("Train", id="btn-train", n_clicks=0, color="primary",
                       style={"fontSize": "18px", "padding": "10px 20px",
                              "width": "60%", "display": "block", "margin": "0 auto"}))
        )
    ], title="GNN Parameters")
], style={"border": "1px solid #ddd", "borderRadius": "8px","fontSize": "22px",
          "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})

sidebar = html.Div([dataset_controls, gnn_controls], style=SIDEBAR_STYLE)

# ---------- Content ----------
content = html.Div([
    html.H4("Software Architecture Analysis (GNNs + Graph Auto-Encoders)",
            className="mt-3 mb-4 text-center",
            style={"fontSize": "30px", "color": "#0c1127", "fontWeight": "600", "letterSpacing": "1px"}),
    dbc.Row([
        mini_kpi("Files", "kpi-files"),
        mini_kpi("Dep. Types", "kpi-types"),
        mini_kpi("Edges", "kpi-edges")
    ], className="g-2 justify-content-center mb-3", style={"gap": "20px"}),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Dependency Type Distribution", className="fw-bold text-center",
                           style={"fontSize": "22px", "color": "#0c1127"}),
            dbc.CardBody(dcc.Graph(id="dep-dist", style={"height": "30vh"}))
        ]), md=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Module Distribution", className="fw-bold text-center",
                           style={"fontSize": "22px", "color": "#0c1127"}),
            dbc.CardBody(dcc.Graph(id="mod-dist", style={"height": "30vh"}))
        ]), md=6)
    ], className="g-3 mb-4"),
    html.H5("Architectural Views", className="text-center fw-bold mb-3",
            style={"fontSize": "26px", "color": "#2b3d8f"}),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Implemented Architecture", className="fw-bold text-center",
                           style={"fontSize": "22px", "color": "#0c1127"}),
            dbc.CardBody(dcc.Loading(dcc.Graph(id="implemented-arch", style={"height": "30vh"}), type="dot"))
        ]), md=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Intended Architecture", className="fw-bold text-center",
                           style={"fontSize": "22px", "color": "#0c1127"}),
            dbc.CardBody(dcc.Loading(dcc.Graph(id="intended-arch", style={"height": "30vh"}), type="dot"))
        ]), md=6)
    ], className="g-3 mb-4"),
    html.P("Edge thickness represents number of dependencies; arrows indicate direction of dependency.",
           className="text-muted text-center", style={"fontSize": "20px", "fontStyle": "italic"}),
    html.Hr(),
    html.H5("Graph Auto-Encoder (GAE) Analysis", className="text-center fw-bold mb-3",
            style={"fontSize": "30px", "color": "#1a237e"}),
    dcc.Loading(dcc.Graph(
        id="emb-plot",
        figure=px.scatter(title="Train once, then switch UMAP/t-SNE to re-project"),
        style={"height": "70vh"}), type="dot"),

    dcc.Store(id="nodes-store"),
    dcc.Store(id="deps-store"),
    dcc.Store(id="embeddings-store"),
    dcc.Store(id="modules-store"),
    dcc.Store(id="trigger-init", data=True),
    dcc.Interval(id="init-interval", n_intervals=0, interval=500, max_intervals=1)
], style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content])


@app.callback(
    Output("nodes-store", "data"),
    Output("deps-store", "data"),
    Input("dataset", "value")
)
def load_dataset(ds_name):
    """Load dataset files into memory for EDA only."""
    if not ds_name or ds_name not in DATASETS:
        raise PreventUpdate

    nodes_path, deps_path = DATASETS[ds_name]
    nodes_df, deps_df = load_nodes_edges(nodes_path, deps_path)
    return nodes_df.to_dict("records"), deps_df.to_dict("records")


def normalize_dependency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent dependency column names."""
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    if "source" in lower: rename[lower["source"]] = "Source"
    elif "src" in lower: rename[lower["src"]] = "Source"
    elif "from" in lower: rename[lower["from"]] = "Source"

    if "target" in lower: rename[lower["target"]] = "Target"
    elif "dst" in lower: rename[lower["dst"]] = "Target"
    elif "to" in lower: rename[lower["to"]] = "Target"

    if "type" in lower: rename[lower["type"]] = "Type"
    elif "dependencytype" in lower: rename[lower["dependencytype"]] = "Type"
    elif "dependency_type" in lower: rename[lower["dependency_type"]] = "Type"

    if "weight" in lower: rename[lower["weight"]] = "Weight"
    elif "count" in lower: rename[lower["count"]] = "Weight"

    df = df.rename(columns=rename)
    if "Type" not in df.columns:
        df["Type"] = "dependency"
    if "Weight" not in df.columns:
        df["Weight"] = 1
    return df.groupby(["Source", "Target", "Type"], as_index=False)["Weight"].sum()


@app.callback(
    Output("dep-types", "options"),
    Output("dep-types", "value"),
    Input("deps-store", "data"),
    prevent_initial_call=False
)
def populate_dep_types(deps_data):
    """Fill dependency type dropdown."""
    if not deps_data:
        return [], []
    df = normalize_dependency_columns(pd.DataFrame(deps_data))
    types = sorted(df["Type"].unique())
    options = [{"label": t, "value": t} for t in types]
    return options, types

@app.callback(
    Output("kpi-files", "children"),
    Output("kpi-types", "children"),
    Output("kpi-edges", "children"),
    Output("dep-dist", "figure"),
    Output("mod-dist", "figure"),
    Output("implemented-arch", "figure"),
    Output("intended-arch", "figure"),
    Input("trigger-init", "data"),
    Input("nodes-store", "data"),
    Input("deps-store", "data"),
    State("dataset", "value"),
    State("topn", "value"),
    prevent_initial_call=False
)
def build_eda(_trigger, nodes_records, deps_records, dataset_name, topn):
    """Simplified and clean EDA (consistent naming and proper refresh on dataset change)."""
    # --- Handle missing data gracefully ---
    if not nodes_records or not deps_records:
        raise PreventUpdate

    # --- Safe conversion to DataFrames ---
    try:
        if isinstance(nodes_records, dict):
            nodes_df = pd.DataFrame.from_dict(nodes_records)
        else:
            nodes_df = pd.DataFrame(nodes_records)

        if isinstance(deps_records, dict):
            deps_df = pd.DataFrame.from_dict(deps_records)
        else:
            deps_df = pd.DataFrame(deps_records)
    except Exception as e:
        print(f"[ERROR] Could not build DataFrames: {e}")
        raise PreventUpdate

    # --- KPIs ---
    num_files = len(nodes_df)
    num_edges = len(deps_df)
    num_types = deps_df["Dependency_Type"].nunique() if "Dependency_Type" in deps_df.columns else 0

    # --- Dependency distribution ---
    if "Dependency_Type" in deps_df.columns and len(deps_df):
        dep_counts = (
            deps_df["Dependency_Type"]
            .astype(str)
            .value_counts()
            .head(int(topn or 10))
            .sort_values(ascending=True)
            .reset_index()
        )
        dep_counts.columns = ["Dependency_Type", "Count"]
        dep_fig = px.bar(
            dep_counts,
            y="Dependency_Type",
            x="Count",
            orientation="h",
            color_discrete_sequence=["#1f77b4"],
        )
    else:
        dep_fig = px.bar(title="Dependency Type Distribution (Unavailable)")

    dep_fig.update_layout(yaxis_title=None, xaxis_title=None)

    # --- Module distribution ---
    if "Module" in nodes_df.columns and len(nodes_df):
        mod_counts = (
            nodes_df["Module"]
            .astype(str)
            .value_counts()
            .head(int(topn or 10))
            .sort_values(ascending=True)
            .reset_index()
        )
        mod_counts.columns = ["Module", "Count"]
        mod_fig = px.bar(
            mod_counts,
            x="Module",
            y="Count",
            color_discrete_sequence=["#1f77b4"],
        )
    else:
        mod_fig = px.bar(title="Module Distribution (Unavailable)")

    mod_fig.update_layout(yaxis_title=None, xaxis_title=None)

    # --- Implemented architecture ---
    try:
        implemented_fig = plot_implemented_architecture(nodes_df, deps_df)
    except Exception as e:
        print(f"[WARN] Implemented architecture plot failed: {e}")
        implemented_fig = px.scatter(title="Implemented Architecture (Error)")

    # --- Intended architecture ---
    try:
        base_name = Path(DATASETS[dataset_name][0]).stem if dataset_name else ""
        rel_path = DATA_DIR / f"{base_name}_relations.csv"
        if rel_path.exists():
            relations_df = pd.read_csv(rel_path)
            intended_fig = plot_intended_architecture(relations_df)
        else:
            intended_fig = px.scatter(title="No intended architecture file")
    except Exception as e:
        print(f"[WARN] Intended architecture plot failed: {e}")
        intended_fig = px.scatter(title="Intended Architecture (Error)")

    # --- Layout polish ---
    for fig in (dep_fig, mod_fig, implemented_fig, intended_fig):
        fig.update_layout(
            font=dict(size=18),
            title_x=0.5,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

    return num_files, num_types, num_edges, dep_fig, mod_fig, implemented_fig, intended_fig


#####################################
# ---------GAER tRAINIFG ------------
#####################################
@app.callback(
    Output("embeddings-store", "data"),
    Output("modules-store", "data"),
    Output("emb-plot", "figure", allow_duplicate=True),
    Input("btn-train", "n_clicks"),
    State("dataset", "value"),
    State("dep-types", "value"),
    State("feat-mode", "value"),
    State("enc-type", "value"),
    State("model-type", "value"),
    State("hidden-dim", "value"),
    State("epochs", "value"),
    State("proj-method", "value"),
    prevent_initial_call=True
)
def train_embeddings(n_clicks, ds_name, selected_types, feat_mode, enc_type,
                     model_type, hidden_dim, epochs, proj_method):
    """Train GAE model and visualize embeddings."""
    if not n_clicks:
        raise PreventUpdate
    if not DATASETS:
        return None, None, px.scatter(title="No datasets")

    ds_name = ds_name or DEFAULT_DATASET
    if ds_name not in DATASETS:
        ds_name = list(DATASETS.keys())[0]

    nodes_path, deps_path = DATASETS[ds_name]

    # use your correct build_dataset signature
    data, _, _ = build_dataset(nodes_path, deps_path,
                               chosen_types=selected_types or [],
                               feature_type=feat_mode, w2v_dim=100)

    # match your old correct call
    z, logs = train_gae(data, hidden=hidden_dim, epochs=epochs,
                        encoder=enc_type, model=model_type)

    emb2d = project_2d(z, method=proj_method)
    df = pd.DataFrame(emb2d, columns=["x", "y"])
    modules = getattr(data, "module_names", ["?"] * len(df))
    df["Module"] = modules

    uniq = sorted(df["Module"].unique())
    palette = px.colors.qualitative.Bold + px.colors.qualitative.Prism + px.colors.qualitative.Alphabet
    cmap = {m: palette[i % len(palette)] for i, m in enumerate(uniq)}

    fig = px.scatter(
        df, x="x", y="y", color="Module", color_discrete_map=cmap,
        title=f"{proj_method.upper()} Projection – GAE Loss {logs['loss']:.4f}"
    )

    fig.update_traces(marker=dict(size=16, opacity=0.99,
                                  line=dict(width=0.6, color="black")))

    fig.update_layout(
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=26),
        title_x=0.5,
        showlegend=True,
        legend=dict(
            title_font=dict(size=24, color="#0c1127"),
            font=dict(size=22),
            bgcolor="rgba(255,255,255,0.8)"
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   title="", showline=True, linewidth=1.2, linecolor="black"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   title="", showline=True, linewidth=1.2, linecolor="black")
    )

    return z.tolist(), modules, fig


@app.callback(
    Output("trigger-init", "data", allow_duplicate=True),
    Input("init-interval", "n_intervals"),
    prevent_initial_call="initial_duplicate"
)
def trigger_startup(_):
    """Trigger initial EDA build on app startup."""
    return True

# ---------- Run ----------
if __name__=="__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
