from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
# Shared W2V helper
from .w2v import W2VEmbeddingGenerator, STOPWORDS, file_level_vectors

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# IO utilities
# ---------------------------------------------------------------------------
def _read_table(path: PathLike) -> pd.DataFrame:
    """Read CSV or Excel by extension, with graceful fallback."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_excel(p)


def load_nodes_edges(nodes_path: PathLike, deps_path: PathLike) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load node and dependency tables and perform cleanup."""
    nodes = _read_table(nodes_path)
    deps = _read_table(deps_path)

    if "File" not in nodes.columns:
        raise ValueError("nodes file must include a 'File' column.")

    for col, default in {
        "Module": "__none__",
        "Entity": None,
        "Code": "",
        "Folder": ""
    }.items():
        if col not in nodes.columns:
            if col == "Entity":
                nodes[col] = nodes["File"].astype(str)
            else:
                nodes[col] = default

    # Validate dependencies
    need = {"Source", "Target"}
    if not need.issubset(deps.columns):
        raise ValueError("deps file must include 'Source' and 'Target' columns.")

    # Keep only edges whose endpoints exist
    valid = set(nodes["File"].astype(str))
    deps = deps[
        deps["Source"].astype(str).isin(valid)
        & deps["Target"].astype(str).isin(valid)
    ].copy()

    # Clean noisy or duplicate deps
    if "Dependency_Type" in deps.columns:
        deps = deps[~deps["Dependency_Type"].astype(str).str.contains("possible", na=False)]
    dup_cols = ["Source", "Target"] + (["Dependency_Count"] if "Dependency_Count" in deps.columns else [])
    deps = deps.drop_duplicates(subset=dup_cols).reset_index(drop=True)

    nodes = nodes.drop_duplicates(subset=["File"]).reset_index(drop=True)
    return nodes, deps


def available_dependency_types(deps: pd.DataFrame) -> List[str]:
    """Return sorted unique Dependency_Type values (if present)."""
    if "Dependency_Type" not in deps.columns:
        return []
    return (
        deps["Dependency_Type"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def _trim_without_ext(fname: str, lang: str) -> str:
    parts = fname.split(".")
    if lang == "java":
        return ".".join(parts[:-1]) if len(parts) > 1 else fname
    if lang in ("c", "c++", "cpp"):
        return ".".join(parts[:-2]) if len(parts) > 2 else fname
    return ".".join(parts[:-1]) if len(parts) > 1 else fname


def file_location_features(files: List[str], language: str = "Java") -> np.ndarray:
    """CountVectorizer over basenames/paths (extension removed)."""
    bases = [_trim_without_ext(str(f), language.lower()) for f in files]
    vect = CountVectorizer(binary=False)
    return vect.fit_transform(bases).toarray().astype(np.float32)


def code_w2v_features(nodes: pd.DataFrame,
                      dim: int = 100,
                      w2v_params: Optional[dict] = None) -> np.ndarray:
    """Generate W2V-based semantic code embeddings at file level."""
    params = dict(vector_size=dim, window=5, min_count=5, sg=1, epochs=10, max_vocab_size=2000)
    params.update(w2v_params or {})
    gen = W2VEmbeddingGenerator(nodes[["Entity", "Code"]], max_df=0.9, stop_words=STOPWORDS)
    emb_map = gen.generate(**params)
    file_entity_df = nodes[["File", "Entity"]].copy()
    return file_level_vectors(file_entity_df, emb_map)


def build_node_features(nodes: pd.DataFrame,
                        feature_type: str,
                        language: str = "Java",
                        w2v_dim: int = 100) -> np.ndarray:
    
    ft = feature_type.strip().lower()
    files = nodes["File"].astype(str).tolist()

    # --- Option 1: Combined folder + semantic features ---
    if ft == "file_location+code_w2v":
        loc = file_location_features(files, language)
        code = code_w2v_features(nodes, dim=w2v_dim)

        # --- Normalize each modality separately ---
        loc_norm = normalize(loc, norm="l2", axis=1)
        code_norm = normalize(code, norm="l2", axis=1)

        # --- Concatenate normalized vectors ---
        return np.hstack([loc_norm, code_norm])

    # --- Option 2: Simple fiel location ---
    if ft == "simple":
        folder_paths = []
        for f in files:
            f = str(f).replace("\\", "/")
            parts = f.split("/")
            folder_path = "/".join(parts[:-1]) if len(parts) > 1 else f
            folder_paths.append(folder_path)

        vec = CountVectorizer(binary=True)
        X = vec.fit_transform(folder_paths).toarray().astype(np.float32)
        return X

    raise ValueError(f"Unknown feature_type: {feature_type}")

# ---------------------------------------------------------------------------
# Heterogeneous Graph Builder
# ---------------------------------------------------------------------------
def build_heterogeneous_graph(
    nodes: pd.DataFrame,
    deps: pd.DataFrame,
    chosen_types: Iterable[str]
) -> Tuple[HeteroData, Dict[str, int]]:
    """Build a heterogeneous graph of FILE nodes separated by Dependency_Type edge types."""
    data = HeteroData()
    chosen = set(str(t) for t in (chosen_types or []))
    deps_f = deps.copy()
    if "Dependency_Type" in deps.columns and chosen:
        deps_f = deps[deps["Dependency_Type"].astype(str).isin(chosen)]

    # Indexing
    nodes = nodes.copy()
    nodes["File_ID"] = range(len(nodes))
    idx_of: Dict[str, int] = dict(zip(nodes["File"].astype(str), nodes["File_ID"]))

    # Node labels
    mods = nodes["Module"].astype(str)
    uniq = {m: i for i, m in enumerate(sorted(set(mods)))}
    y = torch.tensor([uniq[m] for m in mods], dtype=torch.long)
    data["entity"].y = y
    data.module_names = mods.tolist()
    data.num_classes = len(uniq)

    # Edge relations
    rels = (
        deps_f["Dependency_Type"].dropna().unique().tolist()
        if "Dependency_Type" in deps_f.columns
        else ["Dependency"]
    )
    for dep in rels:
        sub = deps_f[deps_f["Dependency_Type"] == dep] if "Dependency_Type" in deps_f.columns else deps_f
        src = sub["Source"].astype(str).map(idx_of)
        tgt = sub["Target"].astype(str).map(idx_of)
        mask = src.notna() & tgt.notna()
        if mask.sum() == 0:
            continue
        edge_index = torch.tensor(
            [src[mask].astype(int).to_numpy(), tgt[mask].astype(int).to_numpy()],
            dtype=torch.long,
        )
        data["entity", dep, "entity"].edge_index = edge_index
        if "Dependency_Count" in sub.columns:
            w = torch.tensor(sub.loc[mask, "Dependency_Count"].astype(float).to_numpy(), dtype=torch.float)
            w = w / (w.max() + 1e-6)
            data["entity", dep, "entity"].edge_weight = w

    return data, idx_of


# ---------------------------------------------------------------------------
# Unified Dataset Builder
# ---------------------------------------------------------------------------
def build_dataset(
    nodes_path: PathLike,
    deps_path: PathLike,
    chosen_types: Iterable[str],
    feature_type: str,
    language: str = "Java",
    w2v_dim: int = 100,
):
    """
    Load files, build a heterogeneous graph, construct node features,
    and return (PyG HeteroData, nodes_df, index_map).
    """
    nodes, deps = load_nodes_edges(nodes_path, deps_path)
    data, idx_of = build_heterogeneous_graph(nodes, deps, chosen_types)

    # Build node features
    X = build_node_features(
        nodes,
        feature_type=feature_type,
        language=language,
        w2v_dim=w2v_dim,
    )

    # Assign features to 'entity' node type
    data["entity"].x = torch.tensor(X, dtype=torch.float32)
    return data, nodes, idx_of
