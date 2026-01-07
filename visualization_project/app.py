from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np, pandas as pd
import plotly.express as px
from dash.exceptions import PreventUpdate
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import networkx as nx


from compute.data_builder import (
    load_nodes_edges,
    available_dependency_types,
    build_dataset,
    code_w2v_features,
)

from compute.gnn_models import train_gae
from compute.projection import project_2d
from compute.plot_architecture import plot_implemented_architecture, plot_intended_architecture
DATASET_CACHE = {}

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
LABEL_STYLE = {"fontSize": "18px", "fontWeight": "600", "marginTop": "8px", **BASE_FONT}

TT_ENCODER = (
    "Encoder = how the GNN combines info from neighbor files (via dependencies).\n\n"
    "GCN: treats neighbors almost equally (simple node-degree normalization).\n\n"
    "GAT: learns which neighbors matter more (attention mechanism)."
)

TT_MODEL = (
    "GAE / VGAE learn embeddings by reconstructing dependency edges.\n\n"
    "GAE: one fixed embedding per file.\n"
    "VGAE: embedding with uncertainty (more regularized)."
)

TT_DEPS = (
    "Dependency types = which edges are included in GNN message passing.\n"
    "Different choices → different embeddings and clusters."
)

TT_FEATURES = (
    "Features = what information each file starts with.\n\n"
    "File location + Code (W2V): uses folder path + code identifiers.\n"
    "File identifier: uses only the file name/id."
)
TT_HIDDEN = (
    "Size of the learned embedding.\n"
    "Larger → more expressive, but slower and may overfit."
)

def label_with_info(text: str, info_id: str):
    return html.Div(
        [
            html.Span(text, style=LABEL_STYLE),
            html.Span(
                " ⓘ",
                id=info_id,
                style={
                    "cursor": "help",
                    "color": "#0d47a1",
                    "fontWeight": "700",
                    "marginLeft": "6px",
                    "fontSize": "16px",
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center"},
    )
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "22rem",
    "padding": "1.4rem 1.1rem",
    "backgroundColor": "#f8f9fa",
    "overflowY": "auto",
    "borderRight": "1px solid #ddd",
    "fontSize": "16px",
    "lineHeight": "1.6",
    **BASE_FONT,
}


CONTENT_STYLE = {
    "marginLeft": "24rem",        
    "padding": "1rem 1.6rem",
    "backgroundColor": "#fff",
    "minHeight": "100vh",
    **BASE_FONT
}


# def mini_kpi(title, comp_id):
#     return dbc.Col(
#         html.Div(
#             [
#                 html.Div(title, className="text-muted",
#                          style={"fontSize": "20px", "fontWeight": "500"}),
#                 html.Div(id=comp_id, className="fw-bold",
#                          style={"fontSize": "26px", "color": "#0d47a1"})
#             ],
#             className="p-2 text-center",
#             style={"border": "1px solid #ddd", "borderRadius": "8px",
#                    "backgroundColor": "#fff", "minWidth": "110px",
#                    "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}
#         ),
#         width="auto"
#     )

# ------------------------------------------------------
# ----------------------- Side Bar ---------------------
# ------------------------------------------------------
dataset_controls = dbc.Card(
    [
        dbc.CardHeader(html.H4("Dataset", style={
            "textAlign": "center", "fontWeight": "700", "fontSize": "24px", "color": "#1a237e"})),
        dbc.CardBody([
            dcc.Dropdown(id="dataset", options=DATASET_OPTIONS, value=DEFAULT_DATASET, clearable=False,
                         persistence=True, persistence_type="memory", style={"fontSize": "18px"}),
            html.Br(),
            html.P("Top-N Modules", style=LABEL_STYLE),
            dcc.Slider(id="topn", min=5, max=20, step=1, value=15,
                       marks={i: str(i) for i in [5, 10, 15, 20]},
                       tooltip={"always_visible": False, "placement": "bottom"})
        ])
    ],
    style={"marginBottom": "20px", "border": "1px solid #ddd", "borderRadius": "8px",
           "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}
)

gnn_controls = dbc.Accordion([
    dbc.AccordionItem([
        label_with_info("Dependency Types", "tt-deps"),
        dcc.Dropdown(id="dep-types", options=[], value=[], multi=True,
                     placeholder="Select dependency types...", style={"fontSize": "18px"}),
        html.Br(),
        dbc.Row([
            dbc.Col([
                label_with_info("Encoder", "tt-encoder"),
                dbc.RadioItems(id="enc-type",
                               options=[{"label": "GAT", "value": "gat"},
                                        {"label": "GCN", "value": "gcn"}],
                               value="gat", inline=True, style={"fontSize": "18px"})
            ], width=5),
            dbc.Col([
                label_with_info("Model", "tt-model"),
                dbc.RadioItems(id="model-type",
                               options=[{"label": "GAE", "value": "gae"},
                                        {"label": "VGAE", "value": "vgae"}],
                               value="gae", inline=True, style={"fontSize": "18px"})
            ], width=5)
        ]),
        html.Br(),
        label_with_info("Features", "tt-features"),
        dbc.RadioItems(id="feat-mode",
                       options=[{"label": "File location + Code (W2V)", "value": "file_location+code_w2v"},
                                {"label": "File identifier", "value": "simple"}],
                       value="file_location+code_w2v", style={"fontSize": "18px"}),
        html.Br(),
        label_with_info("Embedding Dim", "tt-hidden"),
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
    ], 
        title="GNN Parameters")
    ], 
        style={"border": "1px solid #ddd",
                "borderRadius": "8px",
                "fontSize": "22px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"})

sidebar = html.Div([dataset_controls, gnn_controls], style=SIDEBAR_STYLE)

#########################################################
def mini_kpi(title, comp_id):
    return dbc.Col(
        html.Div(
            [
                html.Div(
                    title,
                    className="text-muted",
                    style={"fontSize": "16px", "fontWeight": "500"},
                ),
                html.Div(
                    id=comp_id,
                    className="fw-bold",
                    style={"fontSize": "20px", "color": "#0d47a1"},
                ),
            ],
            className="text-center",
            style={
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "backgroundColor": "#fff",
                "minWidth": "90px",
                "padding": "0.35rem 0.6rem",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
            },
        ),
        width="auto",
    )

# ------------------------------------------------------
# ----------------------- Content ----------------------
# ------------------------------------------------------
content = html.Div(
    [
        dbc.Container(
            [
                # ---------- TITLE ----------
                html.H2(
                    "Software Architecture Analysis (GNNs + Graph Auto-Encoders)",
                    className="text-center my-3",
                    style={"fontWeight": "700", "fontSize": "36px"},
                ),

                # ---------- 2×2 GRID: DISTRIBUTIONS + ARCHITECTURE + KPIs ----------
                dbc.Row(
                    [
                        # LEFT COLUMN: both distributions (stacked)
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5(
                                            "Module Distribution (Top-N)",
                                            style={
                                                "fontWeight": "600",
                                                "fontSize": "26px",
                                                "marginBottom": "0.4rem",
                                                "textAlign": "center",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="mod-dist",
                                            style={"height": "300px"},
                                        ),
                                        html.Hr(style={"margin": "0.6rem 0"}),
                                        html.H5(
                                            "Dependency Type Distribution",
                                            style={
                                                "fontWeight": "600",
                                                "fontSize": "26px",
                                                "marginBottom": "0.4rem",
                                                "textAlign": "center",
                                            },
                                        ),
                                        dcc.Graph(
                                            id="dep-dist",
                                            style={"height": "300px"},
                                        ),
                                    ],
                                    # make this card body fill full column height
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "height": "100%",
                                    },
                                ),
                                className="mb-4",
                            ),
                            width=6,
                        ),

                        # RIGHT COLUMN: KPIs + Implemented architecture
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    dbc.Row(
                                        [
                                            # LEFT COLUMN — KPIs
                                            dbc.Col(
                                                [
                                                    mini_kpi("Files", "kpi-files"),
                                                    html.Br(),
                                                    mini_kpi("Dep. Types", "kpi-types"),
                                                    html.Br(),
                                                    mini_kpi("Edges", "kpi-edges"),
                                                ],
                                                width=2,
                                                style={
                                                    "padding": "0.2rem",
                                                    "textAlign": "center",
                                                },
                                            ),

                                            # RIGHT COLUMN — Architecture + Explanation
                                            dbc.Col(
                                                [
                                                    html.H5(
                                                        "Implemented Architecture",
                                                        style={
                                                            "fontWeight": "600",
                                                            "fontSize": "26px",
                                                            "textAlign": "center",
                                                            "marginBottom": "0.3rem",
                                                        },
                                                    ),

                                                    # << NEW EXPLANATION TEXT >>
                                                    html.P(
                                                        "Edge thickness indicates the number of dependencies between modules.",
                                                        style={
                                                            "textAlign": "center",
                                                            "fontSize": "16px",
                                                            "color": "#555",
                                                            "marginBottom": "0.8rem",
                                                            "marginTop": "-0.3rem",
                                                        }
                                                    ),

                                                    dcc.Graph(
                                                        id="implemented-arch",
                                                        style={"height": "600px"},
                                                    ),
                                                ],
                                                width=9,
                                            ),
                                        ],
                                        className="g-1",
                                    )
                                ),
                                className="mb-4",
                            ),
                            width=6,
                        ),

                    ],
                    className="mb-3 g-3",
                ),

                # ---------- EMBEDDING PLOT ----------
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(id="emb-plot", style={"height": "600px"})
                    ),
                    className="mb-3",
                ),

                # ---------- HIDDEN STORAGE ----------
                dcc.Store(id="nodes-store"),
                dcc.Store(id="deps-store"),
                dcc.Store(id="degree-store"),
                dcc.Store(id="edges-store"),
                dcc.Store(id="embeddings-store"),
                dcc.Store(id="modules-store"),
                dcc.Store(id="trigger-init", data=True),

                dcc.Interval(
                    id="init-interval",
                    n_intervals=0,
                    interval=500,
                    max_intervals=1,
                ),
            ],
            fluid=True,
        )
    ],
    style={
        "marginLeft": "22rem",        # space for sidebar
        "padding": "0.5rem 0.75rem",
        "maxWidth": "calc(100% - 22rem)",
        "overflowX": "hidden",
    },
)


# ------------------------------------------------
# ------------------- Layout ---------------------
# ------------------------------------------------

app.layout = html.Div(
    [
        sidebar,
        content,

        dbc.Tooltip(
            TT_DEPS,
            target="tt-deps",
            placement="right",
            style={"whiteSpace": "pre-line", "fontSize": "14px", "maxWidth": "320px"},
        ),
        
        dbc.Tooltip(
            TT_ENCODER,
            target="tt-encoder",
            placement="right",
            style={"whiteSpace": "pre-line", "fontSize": "14px", "maxWidth": "320px"},
        ),
        dbc.Tooltip(
            TT_MODEL,
            target="tt-model",
            placement="right",
            style={"whiteSpace": "pre-line", "fontSize": "14px", "maxWidth": "320px"},
        ),
        dbc.Tooltip(
            TT_FEATURES,
            target="tt-features",
            placement="right",
            style={"whiteSpace": "pre-line", "fontSize": "14px", "maxWidth": "320px"},
        ),
        dbc.Tooltip(
            TT_HIDDEN,
            target="tt-hidden",
            placement="right",
            style={"whiteSpace": "pre-line", "fontSize": "14px", "maxWidth": "280px"},
        ),

    ]
)

@app.callback(
    Output("nodes-store", "data"),
    Output("deps-store", "data"),
    Output("degree-store", "data"),
    Output("edges-store", "data"),
    Input("dataset", "value")
)
def load_dataset(ds_name):
    if not ds_name or ds_name not in DATASETS:
        raise PreventUpdate

    nodes_path, deps_path = DATASETS[ds_name]
    nodes_df, deps_df = load_nodes_edges(nodes_path, deps_path)

    if "Module" in nodes_df.columns:
        nodes_df["Module"] = (
            nodes_df["Module"]
            .astype(str)
            .apply(lambda m: m.split(".")[-1])
            .apply(lambda m: "GUI" if m.lower()=="gui" else
                            "CLI" if m.lower()=="cli" else
                            m.capitalize())
        )

    node_ids = nodes_df["File"].astype(str).tolist()
    deps_norm = normalize_dependency_columns(deps_df.copy())
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)

    edges = list(zip(deps_norm["Source"].astype(str),
                     deps_norm["Target"].astype(str)))

    G.add_edges_from(edges)
    degree    = dict(G.degree())
    indegree  = dict(G.in_degree())
    outdegree = dict(G.out_degree())
    degree_data = {
        "degree":     [degree.get(f, 0) for f in node_ids],
        "in_degree":  [indegree.get(f, 0) for f in node_ids],
        "out_degree": [outdegree.get(f, 0) for f in node_ids],
    }

    return (
        nodes_df.to_dict("records"),
        deps_df.to_dict("records"),
        degree_data,
        edges
    )


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

# ----------- EDA ---------------
@app.callback(
    Output("kpi-files", "children"),
    Output("kpi-types", "children"),
    Output("kpi-edges", "children"),
    Output("dep-dist", "figure"),
    Output("mod-dist", "figure"),
    Output("implemented-arch", "figure"),
    Input("nodes-store", "data"),
    Input("deps-store", "data"),
    Input("topn", "value"),
    State("dataset", "value"),
)

def build_eda(nodes_records, deps_records, topn, dataset_name):
    if not nodes_records or not deps_records:
        raise PreventUpdate

    nodes_df = pd.DataFrame(nodes_records)
    deps_df  = pd.DataFrame(deps_records)

    # KPIs
    num_files = len(nodes_df)
    num_edges = len(deps_df)
    num_types = deps_df["Dependency_Type"].nunique() if "Dependency_Type" in deps_df.columns else 0

    # ---------- DEPENDENCY TYPE DISTRIBUTION ----------
    if "Dependency_Type" in deps_df.columns and len(deps_df):

        # Raw counts
        dep_counts = (
            deps_df["Dependency_Type"]
            .astype(str)
            .value_counts()
            .rename_axis("Dependency_Type")
            .reset_index(name="Count")
        )

        # Get all types (even missing later)
        all_types = sorted(deps_df["Dependency_Type"].astype(str).unique())

        # Ensure all appear, then sort by Count DESCENDING
        dep_counts = (
            dep_counts.set_index("Dependency_Type")
            .reindex(all_types, fill_value=0)
            .reset_index()
            .sort_values("Count", ascending=True)   # ascending=True -> largest at bottom (horizontal bar)
        )

        dep_fig = px.bar(
            dep_counts,
            x="Count",
            y="Dependency_Type",
            orientation="h",
            color_discrete_sequence=["#4C78A8"],
        )

        # Update axes + layout
        dep_fig.update_yaxes(
            title=None,
            automargin=True,
            tickfont=dict(size=16),
            categoryorder="array",
            categoryarray=dep_counts["Dependency_Type"].tolist(),   # exact correct order
        )

        dep_fig.update_layout(
            xaxis_title=None,
            margin=dict(l=180, r=20, t=40, b=20),
        )

    else:
        dep_fig = px.bar(title="Dependency Type Distribution (Unavailable)")


    # ---------- MODULE DISTRIBUTION (TOP-N APPLIED HERE ONLY) ----------
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
            color_discrete_sequence=["#4C78A8"],
        )

        mod_fig.update_layout(yaxis_title=None, xaxis_title=None)
        top_modules = set(mod_counts["Module"].astype(str).tolist())
        nodes_top = nodes_df[nodes_df["Module"].astype(str).isin(top_modules)]
        valid_files = set(nodes_top["File"].astype(str).tolist())
        deps_top = deps_df[
            deps_df["Source"].astype(str).isin(valid_files)
            & deps_df["Target"].astype(str).isin(valid_files)
        ]
    else:
        mod_fig = px.bar(title="Module Distribution (Unavailable)")
        nodes_top = nodes_df
        deps_top   = deps_df

    # ---------- IMPLEMENTED ARCHITECTURE ----------
    try:
        implemented_fig = plot_implemented_architecture(nodes_top, deps_top)
    except:
        implemented_fig = px.scatter(title="Implemented Architecture (Error)")

    for fig in (dep_fig, mod_fig, implemented_fig):
        fig.update_layout(
            font=dict(size=18),
            title_x=0.5,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

    return num_files, num_types, num_edges, dep_fig, mod_fig, implemented_fig



# -----------------------------------------------------------
# -------------------- GAER TRAINING ------------------------
# -----------------------------------------------------------
# -----------------------------------------------------------
# ----------- INITIAL W2V PROJECTION ON STARTUP ------------
# -----------------------------------------------------------
@app.callback(
    Output("emb-plot", "figure"),
    Input("nodes-store", "data"),
    Input("proj-method", "value"),      # <- proj-method is now an INPUT
    State("degree-store", "data"),
    prevent_initial_call=False,
)
def init_w2v_projection(nodes_records, proj_method, degree_store):
    """Show W2V-only projection automatically when a dataset is loaded."""
    if not nodes_records:
        raise PreventUpdate

    # Fallback: if proj_method is None for some reason, use tsne
    proj_method = proj_method or "tsne"

    nodes_df = pd.DataFrame(nodes_records)

    # 1) compute code-only W2V features
    code_vecs = code_w2v_features(nodes_df, dim=100)

    # 2) project directly to 2D
    emb2d = project_2d(code_vecs, method=proj_method)
    df = pd.DataFrame(emb2d, columns=["x", "y"])

    # 3) modules for colouring
    if "Module" in nodes_df.columns:
        modules = nodes_df["Module"].astype(str).tolist()
    else:
        modules = ["?"] * len(df)
    df["Module"] = modules

    # 4) degree info (optional)
    n = len(df)
    if degree_store is None:
        df["degree"] = 1
        df["in_degree"] = 0
        df["out_degree"] = 0
    else:
        df["degree"]     = degree_store.get("degree",     [1] * n)
        df["in_degree"]  = degree_store.get("in_degree",  [0] * n)
        df["out_degree"] = degree_store.get("out_degree", [0] * n)

    df["in_degree"]  = pd.to_numeric(df["in_degree"], errors="coerce").fillna(0)
    df["out_degree"] = pd.to_numeric(df["out_degree"], errors="coerce").fillna(0)

    # 5) colour map
    uniq = sorted(df["Module"].unique())
    num_classes = len(uniq)
    if num_classes < 10:
        base_palette = px.colors.qualitative.T10
    else:
        base_palette = px.colors.qualitative.Prism + px.colors.qualitative.Pastel
    palette = [base_palette[i % len(base_palette)] for i in range(num_classes)]
    cmap = {m: palette[i] for i, m in enumerate(uniq)}

    # 6) node size
    df["node_size"] = 6 + df["in_degree"]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Module",
        size="node_size",
        hover_data={
            "Module": True,
            "in_degree": True,
            "out_degree": True,
            "node_size":False,
            "x": False,
            "y": False,
        },
        color_discrete_map=cmap,
        title=f"{proj_method.upper()} Projection – Code W2V only (no dependencies)",
        size_max=40,
    )

    fig.update_layout(
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=20),
        title_x=0.5,
        showlegend=True,
        legend=dict(
            title_text="",
            orientation="v",
            x=1.02,
            xanchor="left",
            y=0.5,
            yanchor="middle",
            font=dict(size=18),
            bgcolor="rgba(255,255,255,0.90)",
        ),
        hoverlabel=dict(font_size=20),
        margin=dict(l=40, r=140, t=60, b=40),
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="black")),
        selector=dict(mode="markers"),
    )

    return fig


# -----------------------------------------------------------
# -------------------- GAER TRAINING ------------------------
# -----------------------------------------------------------
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
    State("degree-store", "data"),
    State("edges-store", "data"),
    prevent_initial_call=True
)
def train_embeddings(n_clicks, ds_name, selected_types, feat_mode, enc_type,
                     model_type, hidden_dim, epochs, proj_method,
                     degree_store, edges_store):

    if not n_clicks:
        raise PreventUpdate
    if not DATASETS:
        return None, None, px.scatter(title="No datasets")

    proj_method = proj_method or "tsne"

    # choose dataset name safely
    ds_name = ds_name or DEFAULT_DATASET
    if ds_name not in DATASETS:
        ds_name = list(DATASETS.keys())[0]
    nodes_path, deps_path = DATASETS[ds_name]

    # --------------------------------------------------
    # CASE A: No dependency types selected  ->  W2V-only
    # --------------------------------------------------
    if not selected_types:
        nodes_df, _ = load_nodes_edges(nodes_path, deps_path)
        code_vecs = code_w2v_features(nodes_df, dim=100)
        emb2d = project_2d(code_vecs, method=proj_method)
        df = pd.DataFrame(emb2d, columns=["x", "y"])

        if "Module" in nodes_df.columns:
            modules = nodes_df["Module"].astype(str).tolist()
        else:
            modules = ["?"] * len(df)
        df["Module"] = modules

        n = len(df)
        if degree_store is None:
            df["degree"] = 1
            df["in_degree"] = 0
            df["out_degree"] = 0
        else:
            df["degree"]     = degree_store.get("degree",     [1] * n)
            df["in_degree"]  = degree_store.get("in_degree",  [0] * n)
            df["out_degree"] = degree_store.get("out_degree", [0] * n)

        df["in_degree"]  = pd.to_numeric(df["in_degree"], errors="coerce").fillna(0)
        df["out_degree"] = pd.to_numeric(df["out_degree"], errors="coerce").fillna(0)

        uniq = sorted(df["Module"].unique())
        num_classes = len(uniq)
        if num_classes < 10:
            base_palette = px.colors.qualitative.T10
        else:
            base_palette = px.colors.qualitative.Prism + px.colors.qualitative.Pastel
        palette = [base_palette[i % len(base_palette)] for i in range(num_classes)]
        cmap = {m: palette[i] for i, m in enumerate(uniq)}

        df["node_size"] = 6 + df["in_degree"]

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="Module",
            size="node_size",
            hover_data={
                "Module": True,
                "in_degree": True,
                "out_degree": True,
                "x": False,
                "y": False,
            },
            color_discrete_map=cmap,
            title=f"{proj_method.upper()} Projection – RAW Textual data without Training",
            size_max=40,
        )

        fig.update_layout(
            height=600,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=20),
            title_x=0.5,
            showlegend=True,
            legend=dict(
                title_text="",
                orientation="v",
                x=1.02,
                xanchor="left",
                y=0.5,
                yanchor="middle",
                font=dict(size=18),
                bgcolor="rgba(255,255,255,0.90)",
            ),
            hoverlabel=dict(font_size=20),
            margin=dict(l=40, r=140, t=60, b=40),
        )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        fig.update_traces(
            marker=dict(line=dict(width=0.5, color="black")),
            selector=dict(mode="markers"),
        )

        return code_vecs.tolist(), modules, fig

    # --------------------------------------------------
    # CASE B: At least one dependency type selected
    # --------------------------------------------------
    data, _, _ = build_dataset(
        nodes_path, deps_path,
        chosen_types=selected_types or [],
        feature_type=feat_mode,
        w2v_dim=100,
    )

    z, logs = train_gae(
        data, hidden=hidden_dim, epochs=epochs,
        encoder=enc_type, model=model_type,
    )

    emb2d = project_2d(z, method=proj_method)
    df = pd.DataFrame(emb2d, columns=["x", "y"])
    modules = getattr(data, "module_names", ["?"] * len(df))
    df["Module"] = modules

    n = len(df)
    if degree_store is None:
        df["degree"] = 1
        df["in_degree"] = 0
        df["out_degree"] = 0
    else:
        df["degree"]     = degree_store.get("degree",     [1] * n)
        df["in_degree"]  = degree_store.get("in_degree",  [0] * n)
        df["out_degree"] = degree_store.get("out_degree", [0] * n)

    df["in_degree"]  = pd.to_numeric(df["in_degree"], errors="coerce").fillna(0)
    df["out_degree"] = pd.to_numeric(df["out_degree"], errors="coerce").fillna(0)

    uniq = sorted(df["Module"].unique())
    num_classes = len(uniq)
    if num_classes < 10:
        base_palette = px.colors.qualitative.T10
    else:
        base_palette = px.colors.qualitative.Prism + px.colors.qualitative.Pastel
    palette = [base_palette[i % len(base_palette)] for i in range(num_classes)]
    cmap = {m: palette[i] for i, m in enumerate(uniq)}

    df["node_size"] = 6 + df["in_degree"]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Module",
        size="node_size",
        hover_data={
            "Module": True,
            "in_degree": True,
            "out_degree": True,
            "x": False,
            "y": False,
            "node_size": False,
        },
        color_discrete_map=cmap,
        title=f"{proj_method.upper()} Projection – GAE Loss {logs['loss']:.4f}",
        size_max=40,
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=20),
        title_x=0.5,
        showlegend=True,
        legend=dict(
            title_text="",
            orientation="v",
            x=1.02,
            xanchor="left",
            y=0.5,
            yanchor="middle",
            font=dict(size=18),
            bgcolor="rgba(255,255,255,0.90)",
        ),
        hoverlabel=dict(font_size=20),
        margin=dict(l=40, r=140, t=60, b=40),
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="black")),
        selector=dict(mode="markers"),
    )

    return z.tolist(), modules, fig

# ----------------- CALLBACKS -------------------------
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
