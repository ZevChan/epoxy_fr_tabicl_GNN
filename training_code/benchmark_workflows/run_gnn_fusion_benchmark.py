from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, GINEConv, GINConv, SAGEConv, global_max_pool, global_mean_pool


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

TARGET_CONFIG = {
    "LOI": {"dir": "LOI", "target_col": "LOI", "task": "regression"},
    "Tg": {"dir": "Tg", "target_col": "Tg", "task": "regression"},
    "TENSILE": {"dir": "TENSILE", "target_col": "Tensile", "task": "regression"},
    "UL94": {"dir": "94", "target_col": "UL94", "task": "classification"},
}

SMILES_COLS = ["EP_SMILES", "FR_SMILES", "CURING_SMILES"]
BASIC_NODE_DIM = 23
BASIC_EDGE_DIM = 5
CHEM_NODE_DIM = 35
CHEM_EDGE_DIM = 12

PTABLE = Chem.GetPeriodicTable()

# Pauling electronegativity and atomic polarizability values for atoms commonly
# appearing in epoxy, curing-agent, and flame-retardant SMILES strings. Missing
# elements fall back to zero rather than failing graph construction.
ELECTRONEGATIVITY = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}
POLARIZABILITY = {
    1: 0.667, 5: 3.03, 6: 1.76, 7: 1.10, 8: 0.802, 9: 0.557,
    14: 5.38, 15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05, 53: 5.35,
}
ATOMIC_VOLUME = {
    1: 14.1, 5: 4.6, 6: 5.3, 7: 17.3, 8: 14.0, 9: 17.1,
    14: 12.1, 15: 17.0, 16: 15.5, 17: 22.7, 35: 26.5, 53: 32.5,
}


@dataclass
class TrainConfig:
    hidden_dim: int = 64
    system_dim: int = 64
    branch_dim: int = 128
    gnn_layers: int = 2
    dropout: float = 0.30
    readout: str = "mean"
    scheduler: str = "none"
    batch_size: int = 32
    epochs: int = 180
    patience: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4


def find_dataset(target: str) -> Path:
    base = ROOT / TARGET_CONFIG[target]["dir"]
    candidates = sorted(base.glob("EP+FR+CURING_SMILES+*_DATASET_20260414.csv"))
    if not candidates:
        candidates = sorted(base.glob("*DATASET*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No dataset CSV found in {base}")
    return candidates[0]


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
    time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
    if all(c in df.columns for c in temp_cols):
        df["T_max"] = df[temp_cols].max(axis=1).fillna(0)
    if all(c in df.columns for c in time_cols):
        df["t_total"] = df[time_cols].sum(axis=1).fillna(0)
    if all(c in df.columns for c in temp_cols + time_cols):
        df["Q_thermal"] = sum(df[t].fillna(0) * df[h].fillna(0) for t, h in zip(temp_cols, time_cols))
    df = df.drop(columns=[c for c in temp_cols + time_cols if c in df.columns], errors="ignore")

    fr_col = "Flame_retardant_AdditionAmount(wt%)"
    cur_col = "Curing_agent_AdditionAmount(wt%)"
    if fr_col in df.columns and cur_col in df.columns:
        fr = df[fr_col].fillna(0)
        cur = df[cur_col].fillna(0)
        df["EP_wt_fraction"] = (100.0 - fr - cur) / 100.0
        df["FR_wt_fraction"] = fr / 100.0
        df["CURING_wt_fraction"] = cur / 100.0
    return df


def atom_features_basic(atom: Chem.Atom) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
        float(atom.GetExplicitValence()),
        float(atom.GetImplicitValence()),
        float(atom.GetMass() / 100.0),
        float(atom.IsInRing()),
        float(atom.GetHybridization().real),
        float(atom.GetTotalValence()),
        float(atom.GetChiralTag().real),
        float(atom.GetNumExplicitHs()),
        float(atom.GetNumImplicitHs()),
        float(atom.GetIsotope()),
        float(atom.GetMass() > 30),
        float(atom.GetAtomicNum() in [7, 8]),
        float(atom.GetAtomicNum() in [15, 16]),
        float(atom.GetAtomicNum() in [9, 17, 35]),
        float(atom.IsInRingSize(5)),
        float(atom.IsInRingSize(6)),
        float(len(atom.GetNeighbors())),
    ]


def _ptable_value(method_name: str, atomic_num: int, scale: float) -> float:
    try:
        value = float(getattr(PTABLE, method_name)(int(atomic_num)))
    except Exception:
        value = 0.0
    return value / scale if scale else value


def _has_bond_to(atom: Chem.Atom, atomic_num: int, bond_type: Chem.BondType | None = None) -> bool:
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() != atomic_num:
            continue
        if bond_type is None or bond.GetBondType() == bond_type:
            return True
    return False


def atom_features_chem(atom: Chem.Atom) -> list[float]:
    z = atom.GetAtomicNum()
    is_halogen = z in {9, 17, 35, 53}
    return atom_features_basic(atom) + [
        float(z) / 100.0,
        ELECTRONEGATIVITY.get(z, 0.0) / 4.0,
        _ptable_value("GetRcovalent", z, 2.0),
        _ptable_value("GetRvdw", z, 3.0),
        _ptable_value("GetNOuterElecs", z, 8.0),
        _ptable_value("GetDefaultValence", z, 8.0),
        POLARIZABILITY.get(z, 0.0) / 10.0,
        ATOMIC_VOLUME.get(z, 0.0) / 50.0,
        float(z == 15 and _has_bond_to(atom, 8, Chem.BondType.DOUBLE)),
        float(z == 8 and _has_bond_to(atom, 15, Chem.BondType.DOUBLE)),
        float((z in {8, 15}) and _has_bond_to(atom, 15 if z == 8 else 8, Chem.BondType.SINGLE)),
        float(is_halogen and any(n.GetAtomicNum() == 6 for n in atom.GetNeighbors())),
    ]


def atom_features(atom: Chem.Atom, feature_mode: str = "basic") -> list[float]:
    if feature_mode == "chem":
        return atom_features_chem(atom)
    if feature_mode == "basic":
        return atom_features_basic(atom)
    raise ValueError(f"Unknown graph feature mode: {feature_mode}")


def bond_features_basic(bond: Chem.Bond) -> list[float]:
    return [
        float(bond.GetBondTypeAsDouble()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo()),
        float(bond.GetBondType() == Chem.BondType.AROMATIC),
    ]


def bond_features_chem(bond: Chem.Bond) -> list[float]:
    z1 = bond.GetBeginAtom().GetAtomicNum()
    z2 = bond.GetEndAtom().GetAtomicNum()
    pair = {z1, z2}
    return bond_features_basic(bond) + [
        float(bond.GetBondType() == Chem.BondType.SINGLE),
        float(bond.GetBondType() == Chem.BondType.DOUBLE),
        float(bond.GetBondType() == Chem.BondType.TRIPLE),
        float(bond.GetBondType() == Chem.BondType.AROMATIC),
        float(pair == {15, 8}),
        float(pair == {6, 15}),
        float((6 in pair) and bool(pair & {9, 17, 35, 53})),
    ]


def bond_features(bond: Chem.Bond, feature_mode: str = "basic") -> list[float]:
    if feature_mode == "chem":
        return bond_features_chem(bond)
    if feature_mode == "basic":
        return bond_features_basic(bond)
    raise ValueError(f"Unknown graph feature mode: {feature_mode}")


def smiles_to_graph(smiles: str, feature_mode: str = "basic") -> Data:
    node_dim = CHEM_NODE_DIM if feature_mode == "chem" else BASIC_NODE_DIM
    edge_dim = CHEM_EDGE_DIM if feature_mode == "chem" else BASIC_EDGE_DIM
    if pd.isna(smiles):
        return Data(x=torch.zeros((1, node_dim)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, edge_dim)))
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return Data(x=torch.zeros((1, node_dim)), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, edge_dim)))

    x = torch.tensor([atom_features(a, feature_mode) for a in mol.GetAtoms()], dtype=torch.float32)
    edges, attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond, feature_mode)
        edges.extend([[i, j], [j, i]])
        attrs.extend([bf, bf])
    if not edges:
        edges = [[0, 0]]
        attrs = [[0.0] * edge_dim]
    return Data(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
    )


class EpoxyGraphDataset(torch.utils.data.Dataset):
    def __init__(self, indices, ep, fr, cur, macro, wt, y):
        self.indices = list(indices)
        self.ep = ep
        self.fr = fr
        self.cur = cur
        self.macro = torch.tensor(macro, dtype=torch.float32)
        self.wt = torch.tensor(wt, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_i = self.indices[i]
        return self.ep[real_i], self.fr[real_i], self.cur[real_i], self.macro[i], self.wt[i], self.y[i]


def collate_graphs(batch):
    ep, fr, cur, macro, wt, y = zip(*batch)
    return Batch.from_data_list(ep), Batch.from_data_list(fr), Batch.from_data_list(cur), torch.stack(macro), torch.stack(wt), torch.stack(y)


class GraphEncoder(nn.Module):
    def __init__(self, conv_type: str, node_dim: int, edge_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == "gcn":
            self.convs = nn.ModuleList([GCNConv(node_dim, hidden_dim), GCNConv(hidden_dim, out_dim)])
        elif conv_type == "sage":
            self.convs = nn.ModuleList([SAGEConv(node_dim, hidden_dim), SAGEConv(hidden_dim, out_dim)])
        elif conv_type == "gat":
            self.convs = nn.ModuleList([GATConv(node_dim, hidden_dim, heads=2, concat=False), GATConv(hidden_dim, out_dim, heads=2, concat=False)])
        elif conv_type == "gin":
            self.convs = nn.ModuleList(
                [
                    GINConv(nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))),
                    GINConv(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))),
                ]
            )
        elif conv_type == "gine":
            self.convs = nn.ModuleList(
                [
                    GINEConv(nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)), edge_dim=edge_dim),
                    GINEConv(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)), edge_dim=edge_dim),
                ]
            )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if self.conv_type == "gine":
            x = F.relu(self.norm1(self.convs[0](x, edge_index, edge_attr)))
            x = F.relu(self.norm2(self.convs[1](x, edge_index, edge_attr)))
        else:
            x = F.relu(self.norm1(self.convs[0](x, edge_index)))
            x = F.relu(self.norm2(self.convs[1](x, edge_index)))
        return global_mean_pool(x, batch)


class FusionGNN(nn.Module):
    def __init__(self, conv_type: str, fusion: str, macro_dim: int, task: str, cfg: TrainConfig, node_dim: int = BASIC_NODE_DIM, edge_dim: int = BASIC_EDGE_DIM):
        super().__init__()
        self.fusion = fusion
        self.task = task
        self.encoder = GraphEncoder(conv_type, node_dim=node_dim, edge_dim=edge_dim, hidden_dim=cfg.hidden_dim, out_dim=cfg.system_dim)
        self.macro = nn.Sequential(nn.Linear(macro_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.ReLU(), nn.Dropout(0.2))

        if fusion == "graph_only":
            fused_dim = cfg.system_dim * 3
        elif fusion == "concat":
            fused_dim = cfg.system_dim * 3 + cfg.hidden_dim
        elif fusion == "weighted_sum":
            fused_dim = cfg.system_dim + cfg.hidden_dim
        elif fusion == "gated":
            self.gate = nn.Linear(cfg.hidden_dim, 3)
            fused_dim = cfg.system_dim + cfg.hidden_dim
        elif fusion == "attention":
            self.attn = nn.Sequential(nn.Linear(cfg.system_dim + cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(), nn.Linear(cfg.hidden_dim, 1))
            fused_dim = cfg.system_dim + cfg.hidden_dim
        elif fusion == "film":
            self.film = nn.Linear(cfg.hidden_dim, cfg.system_dim * 2)
            fused_dim = cfg.system_dim * 3 + cfg.hidden_dim
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        out_dim = 1
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, ep, fr, cur, macro, wt):
        ep_emb = self.encoder(ep)
        fr_emb = self.encoder(fr)
        cur_emb = self.encoder(cur)
        macro_emb = self.macro(macro)
        stack = torch.stack([ep_emb, fr_emb, cur_emb], dim=1)

        if self.fusion == "graph_only":
            fused = torch.cat([ep_emb, fr_emb, cur_emb], dim=1)
        elif self.fusion == "concat":
            fused = torch.cat([ep_emb, fr_emb, cur_emb, macro_emb], dim=1)
        elif self.fusion == "weighted_sum":
            system_emb = (stack * wt.unsqueeze(-1)).sum(dim=1)
            fused = torch.cat([system_emb, macro_emb], dim=1)
        elif self.fusion == "gated":
            weights = torch.softmax(self.gate(macro_emb), dim=1)
            system_emb = (stack * weights.unsqueeze(-1)).sum(dim=1)
            fused = torch.cat([system_emb, macro_emb], dim=1)
        elif self.fusion == "attention":
            macro_repeat = macro_emb.unsqueeze(1).repeat(1, 3, 1)
            scores = self.attn(torch.cat([stack, macro_repeat], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            system_emb = (stack * weights.unsqueeze(-1)).sum(dim=1)
            fused = torch.cat([system_emb, macro_emb], dim=1)
        elif self.fusion == "film":
            gamma, beta = self.film(macro_emb).chunk(2, dim=1)
            ep_emb = gamma * ep_emb + beta
            fr_emb = gamma * fr_emb + beta
            cur_emb = gamma * cur_emb + beta
            fused = torch.cat([ep_emb, fr_emb, cur_emb, macro_emb], dim=1)
        return self.head(fused).squeeze(-1)


class FlexibleGraphEncoder(nn.Module):
    def __init__(self, conv_type: str, node_dim: int, edge_dim: int, hidden_dim: int, out_dim: int, num_layers: int, readout: str, dropout: float):
        super().__init__()
        self.conv_type = conv_type
        self.readout = readout
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(max(1, num_layers)):
            in_dim = node_dim if layer == 0 else hidden_dim
            if conv_type == "gcn":
                conv = GCNConv(in_dim, hidden_dim)
            elif conv_type == "sage":
                conv = SAGEConv(in_dim, hidden_dim)
            elif conv_type == "gat":
                conv = GATConv(in_dim, hidden_dim, heads=2, concat=False)
            elif conv_type == "gin":
                conv = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            elif conv_type == "gine":
                conv = GINEConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)), edge_dim=edge_dim)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        pooled_dim = hidden_dim * 2 if readout == "mean_max" else hidden_dim
        self.out = nn.Sequential(
            nn.Linear(pooled_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, norm in zip(self.convs, self.norms):
            if self.conv_type == "gine":
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.readout == "mean_max":
            pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        elif self.readout == "mean":
            pooled = global_mean_pool(x, batch)
        elif self.readout == "max":
            pooled = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown readout: {self.readout}")
        return self.out(pooled)


class ChemBalancedFusionGNN(nn.Module):
    """Chemistry-aware GNN with equal-depth graph and descriptor branches.

    The graph and explicit-descriptor pathways are projected to the same latent
    width before fusion, avoiding direct concatenation of a small graph vector
    with a much larger descriptor vector.
    """

    def __init__(self, conv_type: str, fusion: str, macro_dim: int, task: str, cfg: TrainConfig, node_dim: int = CHEM_NODE_DIM, edge_dim: int = CHEM_EDGE_DIM):
        super().__init__()
        self.fusion = fusion
        self.task = task
        self.encoder = FlexibleGraphEncoder(
            conv_type=conv_type,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.system_dim,
            num_layers=cfg.gnn_layers,
            readout=cfg.readout,
            dropout=cfg.dropout,
        )
        branch_dim = cfg.branch_dim
        if fusion in {"graph_only", "concat", "film"}:
            graph_dim = cfg.system_dim * 3 + 3
        elif fusion in {"weighted_sum", "gated", "attention"}:
            graph_dim = cfg.system_dim + 3
        else:
            raise ValueError(f"Unknown chem-balanced fusion: {fusion}")
        self.graph_branch = nn.Sequential(
            nn.Linear(graph_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
        )
        self.macro_branch = nn.Sequential(
            nn.Linear(macro_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
        )
        self.component_gate = nn.Linear(branch_dim, 3)
        self.component_attn = nn.Sequential(
            nn.Linear(cfg.system_dim + branch_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, 1),
        )
        self.component_film = nn.Linear(branch_dim, cfg.system_dim * 2)
        self.branch_gate = nn.Sequential(
            nn.Linear(branch_dim * 2, branch_dim),
            nn.ReLU(),
            nn.Linear(branch_dim, branch_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(branch_dim * 5, branch_dim * 2),
            nn.LayerNorm(branch_dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(branch_dim * 2, branch_dim),
            nn.ReLU(),
            nn.Linear(branch_dim, 1),
        )
        self.graph_only_head = nn.Sequential(
            nn.Linear(branch_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(branch_dim, 1),
        )

    def _graph_raw(self, ep_emb, fr_emb, cur_emb, macro_latent, wt):
        stack = torch.stack([ep_emb, fr_emb, cur_emb], dim=1)
        if self.fusion in {"graph_only", "concat"}:
            return torch.cat([ep_emb, fr_emb, cur_emb, wt], dim=1)
        if self.fusion == "weighted_sum":
            system_emb = (stack * wt.unsqueeze(-1)).sum(dim=1)
            return torch.cat([system_emb, wt], dim=1)
        if self.fusion == "gated":
            weights = torch.softmax(self.component_gate(macro_latent), dim=1)
            system_emb = (stack * weights.unsqueeze(-1)).sum(dim=1)
            return torch.cat([system_emb, weights], dim=1)
        if self.fusion == "attention":
            macro_repeat = macro_latent.unsqueeze(1).repeat(1, 3, 1)
            scores = self.component_attn(torch.cat([stack, macro_repeat], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            system_emb = (stack * weights.unsqueeze(-1)).sum(dim=1)
            return torch.cat([system_emb, weights], dim=1)
        if self.fusion == "film":
            gamma, beta = self.component_film(macro_latent).chunk(2, dim=1)
            ep_emb = gamma * ep_emb + beta
            fr_emb = gamma * fr_emb + beta
            cur_emb = gamma * cur_emb + beta
            return torch.cat([ep_emb, fr_emb, cur_emb, wt], dim=1)
        raise ValueError(f"Unknown chem-balanced fusion: {self.fusion}")

    def forward(self, ep, fr, cur, macro, wt):
        ep_emb = self.encoder(ep)
        fr_emb = self.encoder(fr)
        cur_emb = self.encoder(cur)
        macro_latent = self.macro_branch(macro)
        graph_raw = self._graph_raw(ep_emb, fr_emb, cur_emb, macro_latent, wt)
        graph_latent = self.graph_branch(graph_raw)
        if self.fusion == "graph_only":
            return self.graph_only_head(graph_latent).squeeze(-1)
        gate = self.branch_gate(torch.cat([graph_latent, macro_latent], dim=1))
        mixed = gate * graph_latent + (1.0 - gate) * macro_latent
        fused = torch.cat([
            mixed,
            graph_latent,
            macro_latent,
            torch.abs(graph_latent - macro_latent),
            graph_latent * macro_latent,
        ], dim=1)
        return self.head(fused).squeeze(-1)


def load_arrays(target: str, graph_feature_modes: tuple[str, ...] = ("basic",)):
    cfg = TARGET_CONFIG[target]
    df = pd.read_csv(find_dataset(target))
    df = add_physical_features(df)
    df = df.dropna(subset=[cfg["target_col"], *SMILES_COLS]).reset_index(drop=True)
    if cfg["task"] == "classification":
        df = df[df[cfg["target_col"]].isin([0, 1])].copy()
        y = df[cfg["target_col"]].astype(int).values
    else:
        y = df[cfg["target_col"]].astype(float).values

    exclude = set(SMILES_COLS + [cfg["target_col"]])
    macro_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    macro = df[macro_cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    macro = SimpleImputer(strategy="median").fit_transform(macro).astype(np.float32)

    wt_cols = ["EP_wt_fraction", "FR_wt_fraction", "CURING_wt_fraction"]
    if all(c in df.columns for c in wt_cols):
        wt = df[wt_cols].fillna(0).values.astype(np.float32)
    else:
        wt = np.ones((len(df), 3), dtype=np.float32) / 3.0
    wt = np.clip(wt, 0, None)
    wt = wt / np.clip(wt.sum(axis=1, keepdims=True), 1e-6, None)

    graphs_by_mode = {}
    for mode in dict.fromkeys(graph_feature_modes):
        print(f"Precomputing {mode} molecular graphs...")
        ep = [smiles_to_graph(s, mode) for s in df["EP_SMILES"]]
        fr = [smiles_to_graph(s, mode) for s in df["FR_SMILES"]]
        cur = [smiles_to_graph(s, mode) for s in df["CURING_SMILES"]]
        graphs_by_mode[mode] = (ep, fr, cur)
    return df, y, macro, wt, graphs_by_mode, macro_cols


def evaluate(y_true, pred_raw, task: str) -> dict:
    if task == "classification":
        prob = torch.sigmoid(torch.tensor(pred_raw)).numpy()
        pred = (prob >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_true, prob)
        except Exception:
            auc = np.nan
        return {"AUC": auc, "F1": f1_score(y_true, pred, zero_division=0), "Accuracy": accuracy_score(y_true, pred)}
    return {"R2": r2_score(y_true, pred_raw), "RMSE": math.sqrt(mean_squared_error(y_true, pred_raw)), "MAE": mean_absolute_error(y_true, pred_raw)}


def chem_balanced_config(base_cfg: TrainConfig, preset: str) -> TrainConfig:
    is_quick = base_cfg.epochs <= 50
    if preset == "balanced":
        return replace(
            base_cfg,
            hidden_dim=48 if is_quick else 80,
            system_dim=64 if is_quick else 96,
            branch_dim=96 if is_quick else 160,
            gnn_layers=3,
            dropout=0.25,
            readout="mean_max",
            lr=8e-4,
            scheduler="cosine",
        )
    if preset == "wide":
        return replace(
            base_cfg,
            hidden_dim=64 if is_quick else 96,
            system_dim=80 if is_quick else 128,
            branch_dim=128 if is_quick else 192,
            gnn_layers=3,
            dropout=0.20,
            readout="mean_max",
            lr=7e-4,
            scheduler="cosine",
        )
    if preset == "deep":
        return replace(
            base_cfg,
            hidden_dim=64 if is_quick else 96,
            system_dim=80 if is_quick else 128,
            branch_dim=128 if is_quick else 192,
            gnn_layers=4,
            dropout=0.30,
            readout="mean_max",
            lr=6e-4,
            scheduler="cosine",
        )
    raise ValueError(f"Unknown chem-balanced preset: {preset}")


def train_one_fold(model, train_loader, test_loader, y_test, task: str, device, cfg: TrainConfig):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    best_metric = -np.inf
    best_pred = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for ep, fr, cur, macro, wt, y in train_loader:
            ep, fr, cur = ep.to(device), fr.to(device), cur.to(device)
            macro, wt, y = macro.to(device), wt.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(ep, fr, cur, macro, wt)
            loss = criterion(out, y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for ep, fr, cur, macro, wt, _ in test_loader:
                ep, fr, cur = ep.to(device), fr.to(device), cur.to(device)
                macro, wt = macro.to(device), wt.to(device)
                preds.extend(model(ep, fr, cur, macro, wt).cpu().numpy().tolist())
        metrics = evaluate(y_test, np.asarray(preds), task)
        primary = metrics["AUC"] if task == "classification" else metrics["R2"]
        if scheduler is not None:
            scheduler.step()
        if primary > best_metric:
            best_metric = primary
            best_pred = np.asarray(preds)
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= cfg.patience:
            break
    return best_pred, best_epoch


def run(target: str, quick: bool, convs: list[str] | None, fusions: list[str] | None, chem_presets: list[str] | None):
    task = TARGET_CONFIG[target]["task"]
    base_cfg = TrainConfig(epochs=40 if quick else 180, patience=8 if quick else 25, hidden_dim=32 if quick else 64, system_dim=32 if quick else 64)
    default_convs = ["gcn", "gin", "gine"] if quick else ["gcn", "sage", "gat", "gin", "gine"]
    selected_convs = convs or default_convs
    selected_fusions = fusions or (["concat", "weighted_sum", "attention", "film"] if quick else ["concat", "weighted_sum", "gated", "attention", "film"])
    chem_presets = chem_presets or (["balanced"] if quick else ["balanced", "wide", "deep"])

    model_specs = []
    for conv in selected_convs:
        for fusion in selected_fusions:
            for preset in chem_presets:
                cfg = chem_balanced_config(base_cfg, preset)
                model_specs.append({
                    "name": f"Graph_{conv.upper()}_Fusion_{fusion}_chem_balanced_{preset}",
                    "conv": conv,
                    "fusion": fusion,
                    "preset": preset,
                    "graph_mode": "chem",
                    "family": "chem_aware_balanced_gnn_fusion",
                    "cfg": cfg,
                })

    graph_modes = tuple(dict.fromkeys(spec["graph_mode"] for spec in model_specs))
    _, y, macro, wt, graphs_by_mode, macro_cols = load_arrays(target, graph_modes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task == "classification" else KFold(n_splits=5, shuffle=True, random_state=42)

    output_prefix = f"gnn_fusion_{target}_quick" if quick else f"gnn_fusion_{target}"
    fold_path = OUT / f"{output_prefix}_all_folds.csv"
    oof_path = OUT / f"{output_prefix}_oof_predictions.csv"
    summary_path = OUT / f"{output_prefix}_summary.csv"

    if fold_path.exists() and not quick:
        existing_folds = pd.read_csv(fold_path)
    else:
        existing_folds = pd.DataFrame()
    if oof_path.exists() and not quick:
        existing_oof = pd.read_csv(oof_path)
    else:
        existing_oof = pd.DataFrame()

    all_folds = existing_folds.to_dict("records") if not existing_folds.empty else []
    all_oof = existing_oof.to_dict("records") if not existing_oof.empty else []
    completed = set()
    if not existing_folds.empty and {"Model", "Fold"}.issubset(existing_folds.columns):
        completed = set(zip(existing_folds["Model"].astype(str), existing_folds["Fold"].astype(int)))

    def flush_progress() -> None:
        fold_df = pd.DataFrame(all_folds)
        oof_df = pd.DataFrame(all_oof)
        fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
        oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
        if not fold_df.empty:
            primary = "AUC" if task == "classification" else "R2"
            metrics = ["AUC", "F1", "Accuracy"] if task == "classification" else ["R2", "RMSE", "MAE"]
            summary = fold_df.groupby(["Target", "Model", "Family"], as_index=False).agg({m: ["mean", "std"] for m in metrics})
            summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
            summary = summary.sort_values(f"{primary}_mean", ascending=False)
            summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for spec in model_specs:
        model_name = spec["name"]
        conv = spec["conv"]
        fusion = spec["fusion"]
        cfg = spec["cfg"]
        ep, fr, cur = graphs_by_mode[spec["graph_mode"]]
        node_dim = int(ep[0].x.size(1))
        edge_dim = int(ep[0].edge_attr.size(1))
        print(f"\n[{target}] {model_name}")
        for fold, (tr, te) in enumerate(splitter.split(macro, y), start=1):
            if (model_name, fold) in completed:
                print(f"  Fold {fold}/5 already completed, skipping.")
                continue
            scaler = StandardScaler()
            macro_tr = scaler.fit_transform(macro[tr]).astype(np.float32)
            macro_te = scaler.transform(macro[te]).astype(np.float32)

            train_ds = EpoxyGraphDataset(tr, ep, fr, cur, macro_tr, wt[tr], y[tr])
            test_ds = EpoxyGraphDataset(te, ep, fr, cur, macro_te, wt[te], y[te])
            train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_graphs)
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, collate_fn=collate_graphs)

            model = ChemBalancedFusionGNN(conv, fusion, macro_dim=len(macro_cols), task=task, cfg=cfg, node_dim=node_dim, edge_dim=edge_dim).to(device)
            t0 = time.time()
            pred_raw, best_epoch = train_one_fold(model, train_loader, test_loader, y[te], task, device, cfg)
            metrics = evaluate(y[te], pred_raw, task)
            all_folds.append({"Target": target, "Model": model_name, "Family": spec["family"], "Preset": spec["preset"], "Fold": fold, "Best_Epoch": best_epoch, "Seconds": time.time() - t0, **metrics})
            for idx, raw in zip(te, pred_raw):
                score = float(torch.sigmoid(torch.tensor(raw)).item()) if task == "classification" else float(raw)
                pred = int(score >= 0.5) if task == "classification" else score
                all_oof.append({"Target": target, "Model": model_name, "Index": int(idx), "Actual": y[idx], "Predicted": pred, "Score": score, "Fold": fold})

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            completed.add((model_name, fold))
            flush_progress()

    fold_df = pd.DataFrame(all_folds)
    oof_df = pd.DataFrame(all_oof)
    primary = "AUC" if task == "classification" else "R2"
    metrics = ["AUC", "F1", "Accuracy"] if task == "classification" else ["R2", "RMSE", "MAE"]
    summary = fold_df.groupby(["Target", "Model", "Family"], as_index=False).agg({m: ["mean", "std"] for m in metrics})
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary = summary.sort_values(f"{primary}_mean", ascending=False)

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    with (OUT / f"{output_prefix}_manifest.json").open("w", encoding="utf-8") as f:
        manifest_specs = [
            {k: (v.__dict__ if k == "cfg" else v) for k, v in spec.items()}
            for spec in model_specs
        ]
        json.dump({"target": target, "task": task, "quick": quick, "model_specs": manifest_specs}, f, indent=2)
    print(summary.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=list(TARGET_CONFIG))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--convs", nargs="*", choices=["gcn", "sage", "gat", "gin", "gine"])
    parser.add_argument("--fusions", nargs="*", choices=["graph_only", "concat", "weighted_sum", "gated", "attention", "film"])
    parser.add_argument("--chem-presets", nargs="*", choices=["balanced", "wide", "deep"])
    args = parser.parse_args()
    run(args.target, args.quick, args.convs, args.fusions, args.chem_presets)


if __name__ == "__main__":
    main()
