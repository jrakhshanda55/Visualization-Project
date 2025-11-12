# compute/w2v.py
import os, re
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Optional
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

STOPWORDS = {
    'name','logger','log','class','public','private','protected','void','static','final','return',
    'int','string','boolean','true','false','null','new','get','set','value','data','object','var','args',
    'this','main','system','out','print','println','import','package','extends','implements','throws','try',
    'catch','exception','file','read','write','input','output','create','loader','strict','update','field',
    'default','comment','response','entry','edit','copy','start','button','check','delete','show','begin',
    'double','float','char','interface','enum','goto','super','abstract','synchronized','volatile','finally',
    'throw','case','break','continue','switch','while','for','do','instanceof','assert','const','unknown',
    'clinit','init','<clinit>','<init>','self','getname','setname','getvalue','setvalue','header','footer'
}

IDENT_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
CAMEL_SNAKE_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[0-9]+")

def build_stopwords(extra: Optional[Iterable[str]] = None) -> set:
    return STOPWORDS if not extra else STOPWORDS.union({s.lower() for s in extra})

class W2VEmbeddingGenerator:
    def __init__(self, df: pd.DataFrame, max_df: float = 0.9, stop_words: Optional[Iterable[str]] = None):
        if "Entity" not in df.columns or "Code" not in df.columns:
            raise ValueError("DataFrame must contain 'Entity' and 'Code'")
        self.df = df.copy()
        self.df["Entity"] = self.df["Entity"].astype(str)
        self.df["Code"] = self.df["Code"].fillna("").astype(str)
        self.df.drop_duplicates(subset=["Entity"], inplace=True)
        self.max_df = float(max_df)
        self.stop_words = set(stop_words or [])
        self.tokens_per_entity: Dict[str, List[str]] = {}
        self._build_corpus()

    def _split_identifier(self, text: str) -> List[str]:
        out: List[str] = []
        for piece in text.split("_"):
            for part in CAMEL_SNAKE_SPLIT.findall(piece):
                t = part.lower()
                if len(t) >= 2 and t not in self.stop_words and not t.isdigit():
                    out.append(t)
        return out

    def _tokens_from_code(self, code: str) -> List[str]:
        if not code:
            return []
        toks: List[str] = []
        for ident in IDENT_REGEX.findall(code):
            if ident.isupper():
                continue
            toks.extend(self._split_identifier(ident))
        return toks

    def _build_corpus(self) -> None:
        raw_tokens: Dict[str, List[str]] = {}
        for _, row in self.df.iterrows():
            entity = row["Entity"]
            raw_tokens[entity] = self._tokens_from_code(row["Code"])
        token_counts: Dict[str, int] = {}
        for toks in raw_tokens.values():
            for tok in set(toks):
                token_counts[tok] = token_counts.get(tok, 0) + 1
        num_entities = max(1, len(raw_tokens))
        threshold = self.max_df * num_entities
        allowed = {tok for tok, c in token_counts.items() if c < threshold}
        self.tokens_per_entity = {e: [t for t in toks if t in allowed] for e, toks in raw_tokens.items()}

    def generate(self,
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 5,
                 sg: int = 1,
                 epochs: int = 10,
                 max_vocab_size: Optional[int] = 2000) -> Dict[str, np.ndarray]:
        sentences = list(self.tokens_per_entity.values())
        if not sentences:
            return {}
        workers = max(1, (os.cpu_count() or 2) - 1)
        w2v = Word2Vec(
            sentences=sentences,
            vector_size=vector_size, window=window, min_count=min_count,
            sg=sg, epochs=epochs, workers=workers, max_vocab_size=max_vocab_size
        )
        zero = np.zeros(vector_size, dtype=np.float32)
        embeddings: Dict[str, np.ndarray] = {}
        for entity, toks in self.tokens_per_entity.items():
            vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
            embeddings[entity] = np.mean(vecs, axis=0).astype(np.float32) if vecs else zero
        return embeddings

def file_level_vectors(df_files: pd.DataFrame, emb_map: Dict[str, np.ndarray]) -> np.ndarray:
    # df_files must have columns: File, Entity
    zero = next(iter(emb_map.values())).copy() if emb_map else np.zeros(100, dtype=np.float32)
    grouped = df_files.groupby("File")["Entity"].apply(list)
    rows = []
    for _, ents in grouped.items():
        vecs = [emb_map.get(e) for e in ents if e in emb_map]
        v = np.mean(vecs, axis=0) if vecs else zero
        rows.append(v)
    X = np.stack(rows, axis=0)
    return normalize(X, norm="l2")
