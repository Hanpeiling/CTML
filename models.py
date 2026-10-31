from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from graph_utils import ATOM_FEATURE_DIM, NUM_RELATIONS, GraphBatch, segment_softmax, segment_sum


class RelationalGraphConv(nn.Module):
    

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = NUM_RELATIONS, dropout: float = 0.1):
        super().__init__()
        self.src = nn.Linear(in_dim, out_dim, bias=False)
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.rel_emb = nn.Embedding(num_relations, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        if edge_index.numel() == 0:
            out = self.self_lin(x)
            return self.dropout(F.gelu(self.norm(out + residual)))
        src, dst = edge_index[0], edge_index[1]
        msg = self.src(x[src]) + self.rel_emb(edge_type)
        agg = x.new_zeros((x.shape[0], msg.shape[1]))
        agg.index_add_(0, dst, msg)
        deg = x.new_zeros((x.shape[0], 1))
        deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=x.device, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0)
        out = agg + self.self_lin(x)
        return self.dropout(F.gelu(self.norm(out + residual)))


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        dims = [input_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            [RelationalGraphConv(dims[i], dims[i + 1], dropout=dropout) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x


class GraphAttentionPooling(nn.Module):
    

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, node_batch: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x).squeeze(-1)
        weights = segment_softmax(scores, node_batch, batch_size)
        pooled = segment_sum(x * weights.unsqueeze(-1), node_batch, batch_size)
        return pooled, weights


class TargetConditionedAtomPooling(nn.Module):
    

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        node_batch: torch.Tensor,
        batch_size: int,
        task_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if task_embeddings.ndim != 2 or task_embeddings.shape[1] != x.shape[1]:
            raise ValueError(
                f"Expected task embeddings [T,{x.shape[1]}], got {tuple(task_embeddings.shape)}"
            )
        n_tasks = int(task_embeddings.shape[0])
        atom = x.unsqueeze(1).expand(-1, n_tasks, -1)
        task = task_embeddings.unsqueeze(0).expand(x.shape[0], -1, -1)
        score_in = torch.cat([atom, task, atom * task], dim=-1)
        scores = self.score(score_in).squeeze(-1)  # N x T

        task_pooled = []
        task_weights = []
        for task_id in range(n_tasks):
            w = segment_softmax(scores[:, task_id], node_batch, batch_size)
            pooled = segment_sum(x * w.unsqueeze(-1), node_batch, batch_size)
            task_pooled.append(pooled)
            task_weights.append(w)
        return torch.stack(task_pooled, dim=1), torch.stack(task_weights, dim=1)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max(input_dim, 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HiddenBlock(nn.Module):
    

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureAttentionFusion(nn.Module):
    

    def __init__(self, hidden_dim: int, n_modalities: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * n_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack(list(tokens), dim=1)  # B x M x H
       
        gates = torch.sigmoid(self.score(x).squeeze(-1))  
        scaled = x * (0.5 + gates).unsqueeze(-1)          
        fused = self.proj(scaled.flatten(start_dim=1))
        return fused, gates


class CrossModalInteractionBlock(nn.Module):
   

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, n_modalities: int = 5):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.n_modalities = n_modalities
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        if tokens.ndim != 3 or tokens.shape[1] != self.n_modalities:
            raise ValueError(
                f"Expected B x {self.n_modalities} x H modality tokens, got {tuple(tokens.shape)}"
            )
        updated = []
        
        full_weights = tokens.new_zeros((tokens.shape[0], self.n_modalities, self.n_modalities))
        for i in range(self.n_modalities):
            other_idx = [j for j in range(self.n_modalities) if j != i]
            q = tokens[:, i:i + 1, :]
            kv = tokens[:, other_idx, :]
            out, w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=False)
            h = self.norm1(q + out)
            h = self.norm2(h + self.ffn(h))
            updated.append(h)
            
            mean_w = w.mean(dim=1).squeeze(1)
            full_weights[:, i, other_idx] = mean_w
        return torch.cat(updated, dim=1), full_weights


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TaskSpecificAdapter(nn.Module):
    
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        bottleneck = max(hidden_dim // 2, 16)
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class EarlyFusionRegressor(nn.Module):
    

    def __init__(self, context_dims: Dict[str, int], hidden_dim: int, n_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.node_input = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.graph_encoder = GraphEncoder(hidden_dim, hidden_dim, n_layers, dropout)
        self.pool = GraphAttentionPooling(hidden_dim)
        self.context_order = ["tk", "species", "environment", "endpoint", "measurement"]
        self.context_encoders = nn.ModuleDict(
            {k: MLPEncoder(context_dims[k], hidden_dim, dropout) for k in self.context_order}
        )
        self.fusion = FeatureAttentionFusion(hidden_dim, n_modalities=6, dropout=dropout)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(self, batch: GraphBatch, return_attention: bool = False):
        x = self.node_input(batch.node_features)
        x = self.graph_encoder(x, batch.edge_index, batch.edge_type)
        graph_token, atom_weights = self.pool(x, batch.node_batch, batch.batch_size)
        context_tokens = [self.context_encoders[k](batch.context[k]) for k in self.context_order]
        fused, modality_weights = self.fusion([graph_token] + context_tokens)
        pred = self.head(fused)
        if return_attention:
            return pred, {"modality": modality_weights, "atom": atom_weights}
        return pred


class MidFusionRegressor(nn.Module):
    

    def __init__(self, context_dims: Dict[str, int], hidden_dim: int, n_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.node_input = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.graph_encoder = GraphEncoder(hidden_dim, hidden_dim, n_layers, dropout)
        self.pool = GraphAttentionPooling(hidden_dim)
        self.context_order = ["tk", "species", "environment", "endpoint", "measurement"]
        self.context_encoders = nn.ModuleDict(
            {k: MLPEncoder(context_dims[k], hidden_dim, dropout) for k in self.context_order}
        )
        self.cross_modal = CrossModalInteractionBlock(hidden_dim, num_heads, dropout, n_modalities=6)
        self.post_blocks = nn.ModuleList([HiddenBlock(hidden_dim, dropout) for _ in range(6)])
        self.final_fusion = FeatureAttentionFusion(hidden_dim, n_modalities=6, dropout=dropout)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(self, batch: GraphBatch, return_attention: bool = False):
        x = self.node_input(batch.node_features)
        x = self.graph_encoder(x, batch.edge_index, batch.edge_type)
        graph_token, atom_weights = self.pool(x, batch.node_batch, batch.batch_size)
        context_tokens = [self.context_encoders[k](batch.context[k]) for k in self.context_order]
        hidden_tokens = torch.stack([graph_token] + context_tokens, dim=1)
        interacted, cross_weights = self.cross_modal(hidden_tokens)
        post_tokens = [self.post_blocks[i](interacted[:, i, :]) for i in range(6)]
        fused, modality_weights = self.final_fusion(post_tokens)
        pred = self.head(fused)
        if return_attention:
            return pred, {
                "cross_modal": cross_weights,
                "modality": modality_weights,
                "atom": atom_weights,
            }
        return pred


class LateFusionRegressor(nn.Module):
   
    def __init__(self, context_dims: Dict[str, int], hidden_dim: int, n_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.node_input = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.graph_encoder = GraphEncoder(hidden_dim, hidden_dim, n_layers, dropout)
        self.pool = GraphAttentionPooling(hidden_dim)
        self.structure_head = RegressionHead(hidden_dim, dropout)

        self.context_order = ["tk", "species", "environment", "endpoint", "measurement"]
        self.context_encoders = nn.ModuleDict(
            {k: MLPEncoder(context_dims[k], hidden_dim, dropout) for k in self.context_order}
        )
        self.context_heads = nn.ModuleDict(
            {k: RegressionHead(hidden_dim, dropout) for k in self.context_order}
        )
        
        self.decision_gate = nn.Linear(6, 6)

    def forward(self, batch: GraphBatch, return_attention: bool = False):
        x = self.node_input(batch.node_features)
        x = self.graph_encoder(x, batch.edge_index, batch.edge_type)
        graph_token, atom_weights = self.pool(x, batch.node_batch, batch.batch_size)
        structure_pred = self.structure_head(graph_token)

        modality_predictions = [structure_pred]
        for k in self.context_order:
            h = self.context_encoders[k](batch.context[k])
            modality_predictions.append(self.context_heads[k](h))
        decisions = torch.stack(modality_predictions, dim=1)  # B x 6
        decision_weights = torch.softmax(self.decision_gate(decisions), dim=1)
        decision_weights = 0.90 * decision_weights + 0.10 / decisions.shape[1]
        pred = (decisions * decision_weights).sum(dim=1)
        if return_attention:
            return pred, {
                "decision": decision_weights,
                "submodel_predictions": decisions,
                "atom": atom_weights,
            }
        return pred



class LC50PretrainRegressor(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        species_vocab_size: int,
    ):
        super().__init__()
        self.node_input = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.graph_encoder = GraphEncoder(hidden_dim, hidden_dim, n_layers, dropout)
        self.pool = GraphAttentionPooling(hidden_dim)
        self.lc50_species_embedding = nn.Embedding(max(int(species_vocab_size), 1), hidden_dim)
        self.lc50_time_encoder = MLPEncoder(1, hidden_dim, dropout)
        self.lc50_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.lc50_head = RegressionHead(hidden_dim, dropout)
        nn.init.normal_(self.lc50_species_embedding.weight, std=0.03)

    def forward(self, batch) -> torch.Tensor:
        x = self.node_input(batch.node_features)
        x = self.graph_encoder(x, batch.edge_index, batch.edge_type)
        graph, _ = self.pool(x, batch.node_batch, batch.batch_size)
        species = self.lc50_species_embedding(batch.species_id)
        duration = self.lc50_time_encoder(batch.duration_z.view(-1, 1))
        h = self.lc50_trunk(torch.cat([graph, species, duration], dim=-1))
        return self.lc50_head(h)


class MultiTaskAttentionRegressor(nn.Module):

    def __init__(
        self,
        context_dims: Dict[str, int],
        hidden_dim: int,
        n_layers: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        n_tasks: int = 8,
        transfer_feature_dim: int = 5,
        record_transfer_feature_dim: int = 9,
        lc50_species_vocab_size: int = 1,
        use_lc50_auxiliary: bool = True,
        use_cross_task_evidence: bool = True,
        use_record_level_evidence: bool = True,
        use_task_atom_attention: bool = True,
        use_lc50_expert: bool = True,
        use_fingerprint_expert: bool = True,
        use_evidence_attention: bool = True,
        use_record_quality_gate: bool = True,
        use_task_residual: bool = True,
        record_attention_temperature: float = 0.40,
    ):
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.hidden_dim = int(hidden_dim)
        self.transfer_feature_dim = int(transfer_feature_dim)
        self.record_transfer_feature_dim = int(record_transfer_feature_dim)
        self.use_cross_task_evidence = bool(use_cross_task_evidence)
        self.use_lc50_auxiliary = bool(use_lc50_auxiliary)
        self.use_record_level_evidence = bool(use_record_level_evidence)
        self.use_task_atom_attention = bool(use_task_atom_attention)
        self.use_lc50_expert = bool(use_lc50_expert)
        self.use_fingerprint_expert = bool(use_fingerprint_expert)
        self.use_evidence_attention = bool(use_evidence_attention)
        self.use_record_quality_gate = bool(use_record_quality_gate)
        self.use_task_residual = bool(use_task_residual)
        self.record_attention_temperature = max(float(record_attention_temperature), 0.05)
        self.num_heads_compat = int(num_heads)

        self.node_input = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.graph_encoder = GraphEncoder(hidden_dim, hidden_dim, n_layers, dropout)

       
        self.pool = GraphAttentionPooling(hidden_dim)
        self.task_atom_pool = TargetConditionedAtomPooling(hidden_dim, dropout=dropout)
        self.graph_task_scale_logit = nn.Parameter(torch.full((self.n_tasks,), -1.10))
        self.graph_fusion_norm = nn.LayerNorm(hidden_dim)

        
        self.fingerprint_encoder = MLPEncoder(int(context_dims.get("fingerprint", 1024)), hidden_dim, dropout)
        self.fingerprint_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.fingerprint_gate[-1].weight)
        nn.init.constant_(self.fingerprint_gate[-1].bias, -1.10)

        self.lc50_expert_node_input = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.lc50_expert_graph_encoder = GraphEncoder(hidden_dim, hidden_dim, 5, 0.0)
        self.lc50_expert_pool = GraphAttentionPooling(hidden_dim)
        self.lc50_expert_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.lc50_expert_gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.lc50_expert_gate[-1].weight)
        nn.init.constant_(self.lc50_expert_gate[-1].bias, -1.10)

        self.context_order = ["tk", "species", "environment", "endpoint", "measurement"]
        self.context_encoders = nn.ModuleDict(
            {k: MLPEncoder(context_dims[k], hidden_dim, dropout) for k in self.context_order}
        )

        self.target_task_embedding = nn.Embedding(self.n_tasks, hidden_dim)
        self.source_task_embedding = nn.Embedding(self.n_tasks, hidden_dim)
        nn.init.normal_(self.target_task_embedding.weight, std=0.03)
        nn.init.normal_(self.source_task_embedding.weight, std=0.03)

        
        gate_in = hidden_dim * 3
        self.modality_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.modality_proj = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        
        self.transfer_encoder = nn.Sequential(
            nn.Linear(self.transfer_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.transfer_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.transfer_norm = nn.LayerNorm(hidden_dim)

        
        self.record_transfer_encoder = nn.Sequential(
            nn.Linear(self.record_transfer_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.record_transfer_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.record_transfer_norm = nn.LayerNorm(hidden_dim)
        
        self.record_evidence_task_gate_logit = nn.Parameter(torch.full((self.n_tasks,), -1.10))
        self.record_quality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.record_quality_gate[-1].weight)
        nn.init.zeros_(self.record_quality_gate[-1].bias)
        self.record_source_target_bias = nn.Parameter(torch.zeros(self.n_tasks, self.n_tasks))
        self.record_similarity_strength_raw = nn.Parameter(torch.tensor(0.80))
        self.transfer_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

     
        self.shared_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.shared_head = RegressionHead(hidden_dim, dropout)
        residual_hidden = max(hidden_dim // 2, 16)
        self.task_residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, residual_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(residual_hidden, 1),
            )
            for _ in range(self.n_tasks)
        ])
        self.task_residual_logit = nn.Parameter(torch.full((self.n_tasks,), -1.10))

        
        self.aux_molecular_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.aux_molecular_head = nn.Linear(hidden_dim, 1)

        
        self.lc50_species_embedding = nn.Embedding(max(int(lc50_species_vocab_size), 1), hidden_dim)
        self.lc50_time_encoder = MLPEncoder(1, hidden_dim, dropout)
        self.lc50_trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.lc50_head = RegressionHead(hidden_dim, dropout)
        nn.init.normal_(self.lc50_species_embedding.weight, std=0.03)

        self.task_log_vars = nn.Parameter(torch.zeros(self.n_tasks))

    def uncertainty_weighted_objective(
        self,
        task_index: int,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        s = torch.clamp(self.task_log_vars[int(task_index)], min=-4.0, max=4.0)
        return 0.5 * torch.exp(-s) * base_loss + 0.5 * s

    @torch.no_grad()
    def task_uncertainty_table(self) -> dict[int, dict[str, float]]:
        log_vars = torch.clamp(self.task_log_vars.detach(), min=-4.0, max=4.0)
        raw = torch.exp(-log_vars)
        norm = raw / raw.mean().clamp_min(1e-12)
        residual = torch.sigmoid(self.task_residual_logit.detach())
        graph_scale = (
            torch.sigmoid(self.graph_task_scale_logit.detach())
            if self.use_task_atom_attention
            else torch.zeros_like(self.graph_task_scale_logit.detach())
        )
        record_scale = torch.sigmoid(self.record_evidence_task_gate_logit.detach())
        return {
            i: {
                "log_variance": float(log_vars[i].cpu()),
                "raw_precision_weight": float(raw[i].cpu()),
                "relative_weight": float(norm[i].cpu()),
                "residual_scale": float(residual[i].cpu()),
                "task_atom_scale": float(graph_scale[i].cpu()),
                "record_evidence_scale": float(record_scale[i].cpu()),
            }
            for i in range(self.n_tasks)
        }

    def pcgrad_shared_parameters(self) -> list[torch.nn.Parameter]:
      
        modules = [
            self.node_input,
            self.graph_encoder,
            self.pool,
            self.task_atom_pool,
            self.graph_fusion_norm,
            self.fingerprint_encoder,
            self.fingerprint_gate,
            self.lc50_expert_projection,
            self.lc50_expert_gate,
            self.context_encoders,
            self.modality_gate,
            self.modality_proj,
            self.transfer_encoder,
            self.transfer_gate,
            self.transfer_norm,
            self.record_transfer_encoder,
            self.record_transfer_gate,
            self.record_transfer_norm,
            self.transfer_fusion,
            self.shared_trunk,
            self.shared_head,
        ]
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for module in modules:
            for param in module.parameters():
                if param.requires_grad and id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)
        return params

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        very_negative = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~mask, very_negative)
        weights = torch.softmax(masked, dim=dim)
        any_valid = mask.any(dim=dim, keepdim=True)
        return torch.where(any_valid, weights, torch.zeros_like(weights))

    def forward(self, batch: GraphBatch, return_attention: bool = False):
        
        x = self.node_input(batch.node_features)
        x = self.graph_encoder(x, batch.edge_index, batch.edge_type)

        target_emb = self.target_task_embedding(batch.task_id)
        all_task_emb = self.target_task_embedding.weight

        shared_graph, shared_atom_weights = self.pool(x, batch.node_batch, batch.batch_size)
        if self.use_task_atom_attention:
            task_graph_all, task_atom_weights_all = self.task_atom_pool(
                x, batch.node_batch, batch.batch_size, all_task_emb
            )
            gather_idx = batch.task_id.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
            target_graph = task_graph_all.gather(1, gather_idx).squeeze(1)
            atom_scale_all = torch.sigmoid(self.graph_task_scale_logit)
            target_atom_scale = atom_scale_all[batch.task_id].unsqueeze(-1)
            sublethal_graph = shared_graph + target_atom_scale * target_graph
        else:
            task_graph_all = shared_graph.unsqueeze(1).expand(-1, self.n_tasks, -1)
            task_atom_weights_all = shared_atom_weights.unsqueeze(1).expand(-1, self.n_tasks)
            atom_scale_all = shared_graph.new_zeros((self.n_tasks,))
            sublethal_graph = shared_graph

        
        context_token_map = {k: self.context_encoders[k](batch.context[k]) for k in self.context_order}

        fp_token = sublethal_graph.new_zeros(sublethal_graph.shape)
        fingerprint_scale = sublethal_graph.new_zeros((batch.batch_size, 1))
        if self.use_fingerprint_expert:
            fp_token = self.fingerprint_encoder(batch.context["fingerprint"])
            fp_gate_in = torch.cat([sublethal_graph, fp_token, target_emb], dim=-1)
            fingerprint_scale = torch.sigmoid(self.fingerprint_gate(fp_gate_in))

        lc50_scale = sublethal_graph.new_zeros((batch.batch_size, 1))
        lc50_token = sublethal_graph.new_zeros(sublethal_graph.shape)
        if self.use_lc50_expert:
            lx = self.lc50_expert_node_input(batch.node_features)
            lx = self.lc50_expert_graph_encoder(lx, batch.edge_index, batch.edge_type)
            lc50_raw, _ = self.lc50_expert_pool(lx, batch.node_batch, batch.batch_size)
            lc50_token = self.lc50_expert_projection(lc50_raw)
            lc50_gate_in = torch.cat([
                sublethal_graph, lc50_token, target_emb,
                context_token_map["species"], context_token_map["measurement"],
            ], dim=-1)
            lc50_scale = torch.sigmoid(self.lc50_expert_gate(lc50_gate_in))

        graph_token = self.graph_fusion_norm(
            sublethal_graph + fingerprint_scale * fp_token + lc50_scale * lc50_token
        )

        context_tokens = [context_token_map[k] for k in self.context_order]
        modality_tokens = torch.stack([graph_token] + context_tokens, dim=1)
        target_expand = target_emb.unsqueeze(1).expand(-1, modality_tokens.shape[1], -1)
        modality_gate_input = torch.cat(
            [modality_tokens, target_expand, modality_tokens * target_expand], dim=-1
        )
        modality_scores = self.modality_gate(modality_gate_input).squeeze(-1)
        modality_weights = torch.softmax(modality_scores, dim=1)
        weighted_modalities = modality_tokens * modality_weights.unsqueeze(-1)
        modality_repr = self.modality_proj(weighted_modalities.flatten(start_dim=1))

        transfer_weights = modality_repr.new_zeros((batch.batch_size, self.n_tasks))
        transfer_repr = modality_repr.new_zeros((batch.batch_size, self.hidden_dim))
        if self.use_cross_task_evidence and "transfer" in batch.context:
            evidence = batch.context["transfer"]
            if evidence.ndim != 3 or evidence.shape[1] != self.n_tasks:
                raise ValueError(
                    f"Expected transfer evidence [B,{self.n_tasks},F], got {tuple(evidence.shape)}"
                )
            source_ids = torch.arange(self.n_tasks, device=evidence.device)
            source_emb = self.source_task_embedding(source_ids).unsqueeze(0).expand(batch.batch_size, -1, -1)
            evidence_tokens = self.transfer_encoder(evidence) + source_emb
            target_for_transfer = target_emb.unsqueeze(1).expand(-1, self.n_tasks, -1)
            transfer_gate_input = torch.cat(
                [evidence_tokens, target_for_transfer, evidence_tokens * target_for_transfer], dim=-1
            )
            transfer_scores = self.transfer_gate(transfer_gate_input).squeeze(-1)
            observed = evidence[..., -1] > 0.5
            not_target = source_ids.unsqueeze(0) != batch.task_id.unsqueeze(1)
            valid = observed & not_target
            if self.use_evidence_attention:
                transfer_weights = self._masked_softmax(transfer_scores, valid, dim=1)
            else:
                valid_float = valid.to(evidence_tokens.dtype)
                transfer_weights = valid_float / valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            transfer_repr = torch.sum(evidence_tokens * transfer_weights.unsqueeze(-1), dim=1)
            transfer_repr = self.transfer_norm(transfer_repr)

      
        record_transfer_weights = modality_repr.new_zeros((batch.batch_size, self.n_tasks))
        record_transfer_repr = modality_repr.new_zeros((batch.batch_size, self.hidden_dim))
        record_max_weight = modality_repr.new_zeros((batch.batch_size,))
        record_entropy = modality_repr.new_zeros((batch.batch_size,))
        record_n = modality_repr.new_zeros((batch.batch_size,))
        record_gate_scale = modality_repr.new_zeros((batch.batch_size, 1))
        if (
            self.use_cross_task_evidence
            and self.use_record_level_evidence
            and "transfer_record" in batch.context
            and "transfer_record_task_id" in batch.context
        ):
            record_features = batch.context["transfer_record"]
            record_task_ids = batch.context["transfer_record_task_id"].long()
            if record_features.ndim != 3:
                raise ValueError(
                    f"Expected transfer_record [B,K,F], got {tuple(record_features.shape)}"
                )
            safe_ids = record_task_ids.clamp(min=0, max=self.n_tasks - 1)
            source_emb_r = self.source_task_embedding(safe_ids)
            record_tokens = self.record_transfer_encoder(record_features) + source_emb_r
            target_r = target_emb.unsqueeze(1).expand(-1, record_features.shape[1], -1)
            modality_r = modality_repr.unsqueeze(1).expand(-1, record_features.shape[1], -1)
            record_gate_input = torch.cat(
                [record_tokens, target_r, modality_r, record_tokens * modality_r], dim=-1
            )
            learned_record_score = self.record_transfer_gate(record_gate_input).squeeze(-1)
            similarity_prior = record_features[..., 6].clamp(0.0, 1.0)
            prior_strength = F.softplus(self.record_similarity_strength_raw)
            observed_r = record_features[..., -1] > 0.5
            valid_r = (
                observed_r
                & (record_task_ids >= 0)
                & (record_task_ids != batch.task_id.unsqueeze(1))
            )
            target_idx = batch.task_id.unsqueeze(1).expand_as(safe_ids)
            source_target = self.record_source_target_bias[target_idx, safe_ids]
            record_scores = (
                0.50 * learned_record_score
                + 4.0 * prior_strength * (similarity_prior - 0.5)
                + source_target
            ) / self.record_attention_temperature
            if self.use_evidence_attention:
                record_weights = self._masked_softmax(record_scores, valid_r, dim=1)
            else:
                valid_weight = valid_r.to(record_tokens.dtype)
                record_weights = valid_weight / valid_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            raw_record_repr = torch.sum(record_tokens * record_weights.unsqueeze(-1), dim=1)
            raw_record_repr = self.record_transfer_norm(raw_record_repr)
            valid_float = valid_r.to(modality_repr.dtype)
            record_n = valid_float.sum(dim=1)
            denom = record_n.clamp_min(1.0)
            mean_sim = (similarity_prior * valid_float).sum(dim=1) / denom
            max_sim = similarity_prior.masked_fill(~valid_r, -1e9).max(dim=1).values
            max_sim = torch.where(valid_r.any(dim=1), max_sim, torch.zeros_like(max_sim))
            count_frac = record_n / max(float(record_features.shape[1]), 1.0)
            quality = torch.stack([mean_sim, max_sim, count_frac], dim=1)
            quality_input = torch.cat([modality_repr, target_emb, raw_record_repr, quality], dim=-1)
            if self.use_record_quality_gate:
                gate_logit = (
                    self.record_quality_gate(quality_input).squeeze(-1)
                    + self.record_evidence_task_gate_logit[batch.task_id]
                )
            else:
                
                gate_logit = self.record_evidence_task_gate_logit[batch.task_id]
            record_gate_scale = torch.sigmoid(gate_logit).unsqueeze(-1)
            record_gate_scale = record_gate_scale * valid_r.any(dim=1, keepdim=True).to(record_gate_scale.dtype)
            record_transfer_repr = raw_record_repr * record_gate_scale
            record_max_weight = record_weights.max(dim=1).values
            record_entropy = -(
                record_weights.clamp_min(1e-12)
                * torch.log(record_weights.clamp_min(1e-12))
            ).sum(dim=1)
            record_transfer_weights.scatter_add_(
                1, safe_ids, record_weights * valid_r.to(record_weights.dtype)
            )

        transfer_repr = self.transfer_fusion(
            torch.cat([transfer_repr, record_transfer_repr], dim=-1)
        )

        shared_input = torch.cat([modality_repr, transfer_repr, target_emb], dim=-1)
        shared_repr = self.shared_trunk(shared_input)
        shared_pred = self.shared_head(shared_repr)

        if self.use_task_residual:
            deltas = torch.stack(
                [head(shared_repr).squeeze(-1) for head in self.task_residual_heads], dim=1
            )
            residual_scale = torch.sigmoid(self.task_residual_logit).view(1, -1)
            all_pred = shared_pred.unsqueeze(1) + residual_scale * deltas
        else:
            residual_scale = shared_pred.new_zeros((1, self.n_tasks))
            all_pred = shared_pred.unsqueeze(1).expand(-1, self.n_tasks)
        selected = all_pred.gather(1, batch.task_id.view(-1, 1)).squeeze(1)

      
        shared_expand = shared_graph.unsqueeze(1).expand(-1, self.n_tasks, -1)
        task_embed_expand = all_task_emb.unsqueeze(0).expand(batch.batch_size, -1, -1)
        atom_scale_expand = atom_scale_all.view(1, -1, 1)
        aux_graph = self.graph_fusion_norm(shared_expand + atom_scale_expand * task_graph_all)
        aux_input = torch.cat([aux_graph, shared_expand, task_embed_expand], dim=-1)
        aux_repr = self.aux_molecular_trunk(aux_input)
        aux_pred = self.aux_molecular_head(aux_repr).squeeze(-1)  # B x T

        if return_attention:
            selected_task_atom = task_atom_weights_all.gather(
                1,
                batch.task_id[batch.node_batch].view(-1, 1),
            ).squeeze(1)
            return selected, all_pred, aux_pred, {
                "modality": modality_weights,
                "transfer_source": transfer_weights,
                "record_transfer_source": record_transfer_weights,
                "record_transfer_max_weight": record_max_weight,
                "record_transfer_entropy": record_entropy,
                "record_transfer_n_records": record_n,
                "record_evidence_scale": record_gate_scale.squeeze(-1),
                "record_similarity_prior_strength": prior_strength.expand(batch.batch_size) if (self.use_cross_task_evidence and self.use_record_level_evidence and "transfer_record" in batch.context) else modality_repr.new_zeros((batch.batch_size,)),
                "record_attention_temperature": modality_repr.new_full((batch.batch_size,), float(self.record_attention_temperature)),
                "fingerprint_scale": fingerprint_scale.squeeze(-1),
                "lc50_expert_scale": lc50_scale.squeeze(-1),
                "atom_shared": shared_atom_weights,
                "atom_target": selected_task_atom,
                "residual_scale": residual_scale.expand(batch.batch_size, -1),
                "task_atom_scale": atom_scale_all.view(1, -1).expand(batch.batch_size, -1),
            }
        return selected, all_pred, aux_pred


    def forward_lc50(self, batch) -> torch.Tensor:
        
        if not self.use_lc50_auxiliary:
            raise RuntimeError("LC50 auxiliary branch is disabled")
        x = self.node_input(batch.node_features)
        x = self.graph_encoder(x, batch.edge_index, batch.edge_type)
        graph, _ = self.pool(x, batch.node_batch, batch.batch_size)
        species = self.lc50_species_embedding(batch.species_id)
        duration = self.lc50_time_encoder(batch.duration_z.view(-1, 1))
        h = self.lc50_trunk(torch.cat([graph, species, duration], dim=-1))
        return self.lc50_head(h)


MultiTaskLateFusionRegressor = MultiTaskAttentionRegressor


def build_single_task_model(
    fusion: str,
    context_dims: Dict[str, int],
    hidden_dim: int,
    n_layers: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    fusion = fusion.lower()
    if fusion == "early":
        return EarlyFusionRegressor(context_dims, hidden_dim, n_layers, num_heads, dropout)
    if fusion == "mid":
        return MidFusionRegressor(context_dims, hidden_dim, n_layers, num_heads, dropout)
    if fusion == "late":
        return LateFusionRegressor(context_dims, hidden_dim, n_layers, num_heads, dropout)
    raise ValueError("fusion must be one of: early, mid, late")
