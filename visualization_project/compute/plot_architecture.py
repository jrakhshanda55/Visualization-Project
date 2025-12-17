# import networkx as nx
# import pandas as pd
# import plotly.graph_objects as go
# import math

# # ----- Intnded Architecture ---------
# def normalize_module(m):
#     m = str(m).strip().split(".")[-1].lower()
#     if m == "gui": return "GUI"
#     if m == "cli": return "CLI"
#     return m.capitalize()

# def plot_intended_architecture(relations_df: pd.DataFrame):
#     NODE_SIZE = 38
#     NODE_RADIUS = 0.06
#     ARROW_SIZE = 1.6
#     ARROW_HEAD = 2

#     # --- Column check ---
#     relations_df.columns = [c.strip().lower() for c in relations_df.columns]

#     if not {"source", "target"}.issubset(relations_df.columns):
#         raise ValueError("Relations file must contain 'source' and 'target' columns (case-insensitive).")

#     relations_df["source"] = relations_df["source"].apply(normalize_module)
#     relations_df["target"] = relations_df["target"].apply(normalize_module)

    
#     # --- Build directed graph ---
#     G = nx.DiGraph()
#     for _, row in relations_df.iterrows():
#         src, tgt = str(row["source"]).strip(), str(row["target"]).strip()
#         if src and tgt:
#             G.add_edge(src, tgt)
#     if len(G.nodes()) == 0:
#         raise ValueError("No valid edges found in relations file.")

#     # --- Circular layout (balanced, no squish) ---
#     pos = nx.circular_layout(G, scale=1.0)

#     # --- Edge traces ---
#     edge_traces = []
#     for u, v in G.edges():
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         edge_traces.append(
#             go.Scatter(
#                 x=[x0, x1],
#                 y=[y0, y1],
#                 mode="lines",
#                 line=dict(width=2, color="rgba(60,60,60,0.6)", shape="spline"),
#                 hoverinfo="text",
#                 text=f"{u} → {v}",
#             )
#         )

#     # --- Smart text positioning (no overlap) ---
#     node_x, node_y, node_text, text_positions = [], [], [], []
#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(node)

#         # Compute angle from center → node
#         angle = math.degrees(math.atan2(y, x))
#         if -90 <= angle <= 90:
#             text_positions.append("middle right")  # label placed outside right side
#         else:
#             text_positions.append("middle left")   # label placed outside left side

#     # --- Node markers + text ---
#     node_trace = go.Scatter(
#         x=node_x,
#         y=node_y,
#         mode="markers+text",
#         text=node_text,
#         textposition=text_positions,
#         hoverinfo="text",
#         marker=dict(size=NODE_SIZE, color="#4C78A8", line=dict(width=1, color="black")),
#         textfont=dict(size=22, color="black"),
#     )

#     fig = go.Figure(data=edge_traces + [node_trace])

#     # --- Arrows (directional flow) ---
#     for u, v in G.edges():
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         dx, dy = x1 - x0, y1 - y0
#         dist = math.hypot(dx, dy)
#         if dist == 0:
#             continue
#         ux, uy = dx / dist, dy / dist

#         extra_gap = 0.04
#         x_start = x0 + ux * NODE_RADIUS
#         y_start = y0 + uy * NODE_RADIUS
#         x_end = x1 - ux * (NODE_RADIUS + extra_gap)
#         y_end = y1 - uy * (NODE_RADIUS + extra_gap)

#         fig.add_annotation(
#             ax=x_start,
#             ay=y_start,
#             x=x_end,
#             y=y_end,
#             xref="x",
#             yref="y",
#             axref="x",
#             ayref="y",
#             showarrow=True,
#             arrowhead=ARROW_HEAD,
#             arrowsize=ARROW_SIZE,
#             arrowwidth=2,
#             arrowcolor="rgba(60,60,60,0.8)",
#             opacity=0.9,
#         )

#     # --- Layout  ---
#     fig.update_layout(
#         title='Intended Architecture',
#         showlegend=False,
#         hovermode="closest",
#         autosize=True,
#         margin=dict(l=20, r=40, t=40, b=20),
#         plot_bgcolor="white",
#         paper_bgcolor="white",
#         height=450,
#         font=dict(size=16),  # ← controls axis/legend labels
#         hoverlabel=dict(
#             font_size=22,     # ← this line changes hover text font size
#             font_family="Segoe UI, Roboto, Helvetica, Arial, sans-serif",
#             bgcolor="white",
#             bordercolor="black"
#         )
#     )

#     fig.update_xaxes(visible=False, range=[-1.2, 1.2])
#     fig.update_yaxes(visible=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1)

#     return fig


# # ----------- Implemented ----------
# #####################################
# def plot_implemented_architecture(nodes_df: pd.DataFrame, deps_df: pd.DataFrame):
#     NODE_SIZE = 38
#     NODE_RADIUS = 0.05
#     ARROW_SIZE = 1.6
#     ARROW_HEAD = 2
#     EDGE_MIN_W, EDGE_MAX_W = 1.2, 7

#     # --- Map file → module ---
#     if "Module" in nodes_df.columns:
#         nodes_df["Module"] = (
#             nodes_df["Module"]
#             .astype(str)
#             .apply(lambda m: m.split(".")[-1])
#             .apply(lambda m: "GUI" if m.lower()=="gui" else
#                             "CLI" if m.lower()=="cli" else
#                             m.capitalize())
#         )

#     file_to_module = dict(zip(nodes_df["File"], nodes_df["Module"]))
#     deps_df["Source_Module"] = deps_df["Source"].map(file_to_module)
#     deps_df["Target_Module"] = deps_df["Target"].map(file_to_module)
#     deps_df = deps_df.dropna(subset=["Source_Module", "Target_Module"])
#     deps_df = deps_df[deps_df["Source_Module"] != deps_df["Target_Module"]]

#     # --- Aggregate dependency counts ---
#     mod_edges = (
#         deps_df.groupby(["Source_Module", "Target_Module"])["Dependency_Count"]
#         .sum()
#         .reset_index()
#     )

#     # --- Build directed graph ---
#     G = nx.DiGraph()
#     for _, row in mod_edges.iterrows():
#         G.add_edge(row["Source_Module"], row["Target_Module"], weight=row["Dependency_Count"])

#     # --- Balanced circular layout ---
#     pos = nx.circular_layout(G, scale=1.0)

#     # --- Edge traces with weight-based thickness ---
#     edge_traces = []
#     weights = [d["weight"] for _, _, d in G.edges(data=True)]
#     max_w = max(weights) if weights else 1

#     for u, v, data in G.edges(data=True):
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         w = data["weight"]
#         ratio = w / max_w

#         edge_width = EDGE_MIN_W + (ratio ** 0.8) * (EDGE_MAX_W - EDGE_MIN_W)
#         edge_opacity = 0.3 + ratio * 0.5

#         edge_traces.append(
#             go.Scatter(
#                 x=[x0, x1],
#                 y=[y0, y1],
#                 mode="lines",
#                 line=dict(width=edge_width, color=f"rgba(70,70,70,{edge_opacity})", shape="spline"),
#                 hoverinfo="text",
#                 text=f"{u} → {v}<br>Dependencies: {w}",
#             )
#         )

#     # --- Smart label positioning (no overlap) ---
#     node_x, node_y, node_text, hover_text, text_positions = [], [], [], [], []
#     for node in G.nodes():
#         x, y = pos[node]
#         indeg = G.in_degree(node, weight="weight")
#         outdeg = G.out_degree(node, weight="weight")

#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(node)
#         hover_text.append(f"<b>{node}</b><br>In: {int(indeg)} | Out: {int(outdeg)}")

#         # Place labels to the outside of the circle
#         angle = math.degrees(math.atan2(y, x))
#         if -90 <= angle <= 90:
#             text_positions.append("middle right")
#         else:
#             text_positions.append("middle left")

#     # --- Node scatter ---
#     node_trace = go.Scatter(
#         x=node_x,
#         y=node_y,
#         mode="markers+text",
#         text=node_text,
#         textposition=text_positions,
#         hovertext=hover_text,
#         hoverinfo="text",
#         marker=dict(size=NODE_SIZE, color="#4C78A8", line=dict(width=1, color="black")),
#         textfont=dict(size=22, color="black"),
#     )

#     fig = go.Figure(data=edge_traces + [node_trace])

#     # --- Add directional arrows ---
#     for u, v, data in G.edges(data=True):
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         w = data["weight"]
#         ratio = w / max_w

#         dx, dy = x1 - x0, y1 - y0
#         dist = math.hypot(dx, dy)
#         if dist == 0:
#             continue
#         ux, uy = dx / dist, dy / dist

#         extra_gap = 0.04
#         x_start = x0 + ux * NODE_RADIUS
#         y_start = y0 + uy * NODE_RADIUS
#         x_end = x1 - ux * (NODE_RADIUS + extra_gap)
#         y_end = y1 - uy * (NODE_RADIUS + extra_gap)

#         fig.add_annotation(
#             ax=x_start,
#             ay=y_start,
#             x=x_end,
#             y=y_end,
#             xref="x",
#             yref="y",
#             axref="x",
#             ayref="y",
#             showarrow=True,
#             arrowhead=ARROW_HEAD,
#             arrowsize=ARROW_SIZE,
#             arrowwidth=1.2 + ratio * 1.8,
#             arrowcolor=f"rgba(60,60,60,{0.4 + ratio * 0.4})",
#             opacity=0.85,
#         )

#     # --- Layout (centered, consistent with intended view) ---
#     fig.update_layout(
#         title='Implemenetd Architecture',
#         showlegend=False,
#         hovermode="closest",
#         autosize=True,
#         margin=dict(l=20, r=40, t=40, b=20),
#         plot_bgcolor="white",
#         paper_bgcolor="white",
#         height=450,
#         font=dict(size=16),  # ← controls axis/legend labels
#         hoverlabel=dict(
#             font_size=22,     # ← this line changes hover text font size
#             font_family="Segoe UI, Roboto, Helvetica, Arial, sans-serif",
#             bgcolor="white",
#             bordercolor="black"
#         )
#     )
#     fig.update_xaxes(visible=False, range=[-1.2, 1.2])
#     fig.update_yaxes(visible=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1)

#     return fig


import math
from typing import Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

# ----------------- shared constants -----------------
ARCH_NODE_SIZE = 34
ARCH_NODE_RADIUS = 0.06
ARCH_ARROW_SIZE = 1.6
ARCH_ARROW_HEAD = 2
EDGE_MIN_W, EDGE_MAX_W = 1.2, 7


def normalize_module(m: Optional[str]) -> str:
    """Normalize module labels so GUI/CLI look nice and others are capitalized."""
    m = str(m or "").strip().split(".")[-1].lower()
    if m == "gui":
        return "GUI"
    if m == "cli":
        return "CLI"
    return m.capitalize()


# ----------------- shared plotting helper -----------------
def _build_architecture_figure(
    G: nx.DiGraph,
    title: str,
    use_weights: bool = False,
) -> go.Figure:

    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes.")

    # circular layout
    pos = nx.circular_layout(G, scale=1.0)

    # -------- edge traces ----------
    edge_traces = []
    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1.0

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        w = float(data.get("weight", 1.0))
        if use_weights and max_w > 0:
            ratio = w / max_w
            line_width = EDGE_MIN_W + (ratio ** 0.8) * (EDGE_MAX_W - EDGE_MIN_W)
            edge_opacity = 0.3 + ratio * 0.5
        else:
            line_width = 2
            edge_opacity = 0.6

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=line_width,
                    color=f"rgba(70,70,70,{edge_opacity})",
                    shape="spline",
                ),
                hoverinfo="text",
                text=f"{u} → {v}" + (f"<br>Dependencies: {int(w)}" if use_weights else ""),
            )
        )

    # -------- nodes + labels ----------
    node_x, node_y, node_text, hover_text, text_positions = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        indeg = G.in_degree(node, weight="weight") if use_weights else G.in_degree(node)
        outdeg = G.out_degree(node, weight="weight") if use_weights else G.out_degree(node)
        hover_text.append(f"<b>{node}</b><br>In: {int(indeg)} | Out: {int(outdeg)}")

        angle = math.degrees(math.atan2(y, x))
        text_positions.append("middle right" if -90 <= angle <= 90 else "middle left")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition=text_positions,
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=ARCH_NODE_SIZE,
            color="#4C78A8",
            line=dict(width=1, color="black"),
        ),
        textfont=dict(size=18, color="black"),
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    # -------- arrows ----------
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        w = float(data.get("weight", 1.0))
        if use_weights and max_w > 0:
            ratio = w / max_w
            arrow_width = 1.2 + ratio * 1.8
            arrow_alpha = 0.4 + ratio * 0.4
        else:
            arrow_width = 2.0
            arrow_alpha = 0.8

        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy = dx / dist, dy / dist

        extra_gap = 0.04
        x_start = x0 + ux * ARCH_NODE_RADIUS
        y_start = y0 + uy * ARCH_NODE_RADIUS
        x_end = x1 - ux * (ARCH_NODE_RADIUS + extra_gap)
        y_end = y1 - uy * (ARCH_NODE_RADIUS + extra_gap)

        fig.add_annotation(
            ax=x_start,
            ay=y_start,
            x=x_end,
            y=y_end,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=ARCH_ARROW_HEAD,
            arrowsize=ARCH_ARROW_SIZE,
            arrowwidth=arrow_width,
            arrowcolor=f"rgba(60,60,60,{arrow_alpha})",
            opacity=0.9,
        )

    # -------- layout ----------
    fig.update_layout(
        title=None,
        showlegend=False,
        hovermode="closest",
        autosize=True,
        height=570,  # fits your dashboard card
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=16),
        hoverlabel=dict(
            font_size=18,
            font_family="Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            bgcolor="white",
            bordercolor="black",
        ),
    )
    fig.update_xaxes(visible=False, range=[-1.2, 1.2])
    fig.update_yaxes(visible=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1)

    return fig


# ----------------- public API functions -----------------
def plot_intended_architecture(relations_df: pd.DataFrame) -> go.Figure:
    """High-level wrapper: prepare graph of intended architecture, then plot."""
    df = relations_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if not {"source", "target"}.issubset(df.columns):
        raise ValueError("Relations file must contain 'source' and 'target' columns (case-insensitive).")

    df["source"] = df["source"].apply(normalize_module)
    df["target"] = df["target"].apply(normalize_module)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, tgt = str(row["source"]).strip(), str(row["target"]).strip()
        if src and tgt:
            G.add_edge(src, tgt)

    if G.number_of_nodes() == 0:
        raise ValueError("No valid edges found in relations file.")

    return _build_architecture_figure(G, title=None, use_weights=False)


def plot_implemented_architecture(nodes_df: pd.DataFrame, deps_df: pd.DataFrame) -> go.Figure:
    """High-level wrapper: build module-level graph from code, then plot."""
    nodes = nodes_df.copy()
    deps = deps_df.copy()

    # Normalize module names
    if "Module" in nodes.columns:
        nodes["Module"] = nodes["Module"].apply(normalize_module)

    file_to_module = dict(zip(nodes["File"], nodes["Module"]))
    deps["Source_Module"] = deps["Source"].map(file_to_module)
    deps["Target_Module"] = deps["Target"].map(file_to_module)
    deps = deps.dropna(subset=["Source_Module", "Target_Module"])
    deps = deps[deps["Source_Module"] != deps["Target_Module"]]

    # aggregate dependency counts at module level
    mod_edges = (
        deps.groupby(["Source_Module", "Target_Module"])["Dependency_Count"]
        .sum()
        .reset_index()
    )

    G = nx.DiGraph()
    for _, row in mod_edges.iterrows():
        G.add_edge(row["Source_Module"], row["Target_Module"], weight=row["Dependency_Count"])

    if G.number_of_nodes() == 0:
        raise ValueError("No module-level edges to plot.")

    return _build_architecture_figure(G, title=None, use_weights=True)
