from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, GATConv, GCNConv
from torch_geometric.nn.models import GAE, VGAE
from torch_geometric.data import HeteroData
from torch_geometric.utils import remove_self_loops, to_undirected, coalesce


# ----------------------------------------------------------------------------
# Heterogeneous Graph Encoder
# ----------------------------------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(
        self,
        relations,
        hidden: int = 128,
        dropout: float = 0.0,
        encoder: str = "gat",
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        # keep only ('entity', *, 'entity') relations
        ent_ent = [et for et in relations if et[0] == "entity" and et[2] == "entity"]
        if not ent_ent:
            raise ValueError("No ('entity', *, 'entity') relations found in HeteroData.")

        self.dropout = dropout
        self.encoder = encoder.lower()
        if self.encoder not in {"gat", "gcn"}:
            raise ValueError("encoder must be either 'gat' or 'gcn'")

        def make_conv(out_channels):
            if self.encoder == "gat":
                return HeteroConv(
                    {
                        et: GATConv(
                            (-1, -1),
                            out_channels,
                            heads=1,
                            concat=False,
                            dropout=dropout,
                        )
                        for et in ent_ent
                    },
                    aggr="sum",
                )
            else:  # gcn
                return HeteroConv(
                    {
                        et: GCNConv(
                            -1,
                            out_channels,
                            add_self_loops=add_self_loops,
                            normalize=normalize,
                        )
                        for et in ent_ent
                    },
                    aggr="sum",
                )

        self.conv = make_conv(hidden)

    def forward(self, x_dict, edge_index_dict):
        out = self.conv(x_dict, edge_index_dict)
        out = {k: F.elu(v) for k, v in out.items()}
        out = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in out.items()}
        return out


# ----------------------------------------------------------------------------
# Adapter for GAE
# ----------------------------------------------------------------------------
class _Adapter(nn.Module):
    def __init__(self, enc, for_vgae: bool = False):
        super().__init__()
        self.enc = enc
        self.for_vgae = for_vgae

    def forward(self, x_dict, ei_dict):
        out = self.enc(x_dict, ei_dict)['entity']
        # For VGAE: return μ and logσ
        if self.for_vgae:
            mu = out
            logstd = torch.zeros_like(mu)  # no learned variance yet
            return mu, logstd
        return out

# ----------------------------------------------------------------------------
# Utility: merge all entity-entity edge types
# ----------------------------------------------------------------------------
def _merge_entity_edges(edge_index_dict, num_nodes: int, device: torch.device):
    """Merge all entity→entity relations into a single undirected edge index."""
    parts = [ei for (et, ei) in edge_index_dict.items() if et[0] == "entity" and et[2] == "entity"]
    if not parts:
        raise ValueError("No ('entity', *, 'entity') edge types found in HeteroData.")
    ei = torch.cat(parts, dim=1)
    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei, num_nodes=num_nodes)
    ei = coalesce(ei, num_nodes=num_nodes)
    return ei.to(device)


# ----------------------------------------------------------------------------
# Training routine
# ----------------------------------------------------------------------------
def train_gae(
    data: HeteroData,
    hidden: int = 64,
    dropout: float = 0.0,
    epochs: int = 30,
    lr: float = 1e-3,
    encoder: str = "gat",
    model: str = "gae",            # <--- new
    device: Optional[str] = None,
    add_self_loops: bool = True,
    normalize: bool = True,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    x_dict = {"entity": data["entity"].x.to(device)}
    ei_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}

    # merge entity-entity edges
    parts = [ei for (et, ei) in ei_dict.items() if et[0] == "entity" and et[2] == "entity"]
    if not parts:
        raise ValueError("No ('entity', *, 'entity') edge types found.")
    pos_edge_index = torch.cat(parts, dim=1)
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index = to_undirected(pos_edge_index, num_nodes=data["entity"].num_nodes)
    pos_edge_index = coalesce(pos_edge_index, num_nodes=data["entity"].num_nodes).to(device)

    enc = GraphEncoder(
        relations=data.edge_types,
        hidden=hidden,
        dropout=dropout,
        encoder=encoder,
        add_self_loops=add_self_loops,
        normalize=normalize,
    ).to(device)

    model = model.lower()
    if model == "vgae":
        gae = VGAE(_Adapter(enc, for_vgae=True)).to(device)
    else:  
        gae = GAE(_Adapter(enc)).to(device)
        
    opt = torch.optim.Adam(gae.parameters(), lr=lr)

    last_loss = None
    for _ in range(epochs):
        gae.train()
        opt.zero_grad(set_to_none=True)
        z = gae.encode(x_dict, ei_dict)  # VGAE returns reparam sample too
        loss = gae.recon_loss(z, pos_edge_index)
        # add KL if the model supports it (VGAE)
        if hasattr(gae, "kl_loss"):
            loss = loss + 1e-3 * gae.kl_loss()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    gae.eval()
    with torch.no_grad():
        z = gae.encode(x_dict, ei_dict).detach().cpu().numpy()
    return z, {"loss": last_loss}