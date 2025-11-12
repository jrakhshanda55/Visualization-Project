import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import math

# ----- Intnded Architecture ---------

def plot_intended_architecture(relations_df: pd.DataFrame):
    NODE_SIZE = 38
    NODE_RADIUS = 0.06
    ARROW_SIZE = 1.6
    ARROW_HEAD = 2

    # --- Column check ---
    relations_df.columns = [c.strip().lower() for c in relations_df.columns]
    if not {"source", "target"}.issubset(relations_df.columns):
        raise ValueError("Relations file must contain 'Source' and 'Target' columns.")
    relations_df["source"] = relations_df["source"].str.title().replace({r"(?i)\bgui\b": "GUI", r"(?i)\bcli\b": "CLI"}, regex=True)
    relations_df["target"] = relations_df["target"].str.title().replace({r"(?i)\bgui\b": "GUI", r"(?i)\bcli\b": "CLI"}, regex=True)


    
    # --- Build directed graph ---
    G = nx.DiGraph()
    for _, row in relations_df.iterrows():
        src, tgt = str(row["source"]).strip(), str(row["target"]).strip()
        if src and tgt:
            G.add_edge(src, tgt)
    if len(G.nodes()) == 0:
        raise ValueError("No valid edges found in relations file.")

    # --- Circular layout (balanced, no squish) ---
    pos = nx.circular_layout(G, scale=1.0)

    # --- Edge traces ---
    edge_traces = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=2, color="rgba(60,60,60,0.6)", shape="spline"),
                hoverinfo="text",
                text=f"{u} → {v}",
            )
        )

    # --- Smart text positioning (no overlap) ---
    node_x, node_y, node_text, text_positions = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Compute angle from center → node
        angle = math.degrees(math.atan2(y, x))
        if -90 <= angle <= 90:
            text_positions.append("middle right")  # label placed outside right side
        else:
            text_positions.append("middle left")   # label placed outside left side

    # --- Node markers + text ---
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition=text_positions,
        hoverinfo="text",
        marker=dict(size=NODE_SIZE, color="#1f77b4", line=dict(width=1, color="black")),
        textfont=dict(size=22, color="black"),
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    # --- Arrows (directional flow) ---
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy = dx / dist, dy / dist

        extra_gap = 0.04
        x_start = x0 + ux * NODE_RADIUS
        y_start = y0 + uy * NODE_RADIUS
        x_end = x1 - ux * (NODE_RADIUS + extra_gap)
        y_end = y1 - uy * (NODE_RADIUS + extra_gap)

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
            arrowhead=ARROW_HEAD,
            arrowsize=ARROW_SIZE,
            arrowwidth=2,
            arrowcolor="rgba(60,60,60,0.8)",
            opacity=0.9,
        )

    # --- Layout  ---
    fig.update_layout(
        title=None,
        showlegend=False,
        hovermode="closest",
        autosize=True,
        margin=dict(l=20, r=40, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        font=dict(size=20),  # ← controls axis/legend labels
        hoverlabel=dict(
            font_size=26,     # ← this line changes hover text font size
            font_family="Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            bgcolor="white",
            bordercolor="black"
        )
    )
    fig.update_xaxes(visible=False, range=[-1.2, 1.2])
    fig.update_yaxes(visible=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1)

    return fig


# ----------- Implemented ----------
#####################################
def plot_implemented_architecture(nodes_df: pd.DataFrame, deps_df: pd.DataFrame):
    NODE_SIZE = 38
    NODE_RADIUS = 0.05
    ARROW_SIZE = 1.6
    ARROW_HEAD = 2
    EDGE_MIN_W, EDGE_MAX_W = 1.2, 7

    # --- Map file → module ---
    nodes_df["Module"] = nodes_df["Module"].str.title().replace({"gui": "GUI", "cli": "CLI"}, regex=False)
    nodes_df["Module"] = nodes_df["Module"].str.title()


    file_to_module = dict(zip(nodes_df["File"], nodes_df["Module"]))
    deps_df["Source_Module"] = deps_df["Source"].map(file_to_module)
    deps_df["Target_Module"] = deps_df["Target"].map(file_to_module)
    deps_df = deps_df.dropna(subset=["Source_Module", "Target_Module"])
    deps_df = deps_df[deps_df["Source_Module"] != deps_df["Target_Module"]]

    # --- Aggregate dependency counts ---
    mod_edges = (
        deps_df.groupby(["Source_Module", "Target_Module"])["Dependency_Count"]
        .sum()
        .reset_index()
    )

    # --- Build directed graph ---
    G = nx.DiGraph()
    for _, row in mod_edges.iterrows():
        G.add_edge(row["Source_Module"], row["Target_Module"], weight=row["Dependency_Count"])

    # --- Balanced circular layout ---
    pos = nx.circular_layout(G, scale=1.0)

    # --- Edge traces with weight-based thickness ---
    edge_traces = []
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data["weight"]
        ratio = w / max_w

        edge_width = EDGE_MIN_W + (ratio ** 0.8) * (EDGE_MAX_W - EDGE_MIN_W)
        edge_opacity = 0.3 + ratio * 0.5

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=edge_width, color=f"rgba(70,70,70,{edge_opacity})", shape="spline"),
                hoverinfo="text",
                text=f"{u} → {v}<br>Dependencies: {w}",
            )
        )

    # --- Smart label positioning (no overlap) ---
    node_x, node_y, node_text, hover_text, text_positions = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        indeg = G.in_degree(node, weight="weight")
        outdeg = G.out_degree(node, weight="weight")

        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        hover_text.append(f"<b>{node}</b><br>In: {int(indeg)} | Out: {int(outdeg)}")

        # Place labels to the outside of the circle
        angle = math.degrees(math.atan2(y, x))
        if -90 <= angle <= 90:
            text_positions.append("middle right")
        else:
            text_positions.append("middle left")

    # --- Node scatter ---
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition=text_positions,
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(size=NODE_SIZE, color="#4C78A8", line=dict(width=1, color="black")),
        textfont=dict(size=22, color="black"),
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    # --- Add directional arrows ---
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data["weight"]
        ratio = w / max_w

        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy = dx / dist, dy / dist

        extra_gap = 0.04
        x_start = x0 + ux * NODE_RADIUS
        y_start = y0 + uy * NODE_RADIUS
        x_end = x1 - ux * (NODE_RADIUS + extra_gap)
        y_end = y1 - uy * (NODE_RADIUS + extra_gap)

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
            arrowhead=ARROW_HEAD,
            arrowsize=ARROW_SIZE,
            arrowwidth=1.2 + ratio * 1.8,
            arrowcolor=f"rgba(60,60,60,{0.4 + ratio * 0.4})",
            opacity=0.85,
        )

    # --- Layout (centered, consistent with intended view) ---
    fig.update_layout(
        title=None,
        showlegend=False,
        hovermode="closest",
        autosize=True,
        margin=dict(l=20, r=40, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        font=dict(size=20),  # ← controls axis/legend labels
        hoverlabel=dict(
            font_size=26,     # ← this line changes hover text font size
            font_family="Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            bgcolor="white",
            bordercolor="black"
        )
    )
    fig.update_xaxes(visible=False, range=[-1.2, 1.2])
    fig.update_yaxes(visible=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1)

    return fig