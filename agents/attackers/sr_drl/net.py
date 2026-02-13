from __future__ import annotations

import torch
from torch import nn

EMB_SIZE = 32


def segmented_sample(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample one index according to the given probability vector.
    Expects 1D tensor whose entries sum to 1.
    """
    dist = torch.distributions.Categorical(probs)
    return dist.sample()


class GraphPolicyNet(nn.Module):
    """
    Lightweight graph-based policy + value network inspired by SR-DRL,
    implemented purely in PyTorch (no torch_geometric/torch_scatter).

    - Single graph at a time
    - Single discrete action over provided actionable node indices
    """

    def __init__(
        self,
        num_node_features: int,
        mp_iterations: int = 5,
        lr: float = 3e-3,
        weight_decay: float = 1.0e-4,
        max_grad_norm: float = 3.0,
        device: str | None = None,
    ):
        super().__init__()

        self.mp_iterations = mp_iterations

        # Initial node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(num_node_features, EMB_SIZE),
            nn.LeakyReLU(),
        )

        # Message and update MLPs
        self.msg_mlp = nn.Sequential(
            nn.Linear(EMB_SIZE, EMB_SIZE),
            nn.LeakyReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(EMB_SIZE * 2, EMB_SIZE),
            nn.LeakyReLU(),
        )

        # Value from global pooled embedding
        self.value_head = nn.Linear(EMB_SIZE, 1)

        # Policy over nodes (per-node logits)
        self.policy_head = nn.Linear(EMB_SIZE, 1)

        self.device = torch.device(device or "cpu")
        self.to(self.device)

        self.opt = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.lr = lr
        self.max_grad_norm = max_grad_norm

    def message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simple message passing:
            m_{j->i} = MLP(x_j)
            m_i = sum_j m_{j->i}
            x_i' = U([x_i, m_i])
        """
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index  # (E,), (E,)
        messages = self.msg_mlp(x[src])  # (E, H)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)

        updated = self.update_mlp(torch.cat([x, agg], dim=1))
        return updated

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        actionable_indices: list[int],
        only_v: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_feats: (N, F) node feature tensor
            edge_index: (2, E) edge index
            actionable_indices: indices of nodes that map to actions
            only_v: if True, only compute value

        Returns:
            selected_indices: tensor of shape (1,) with node index
            value: tensor shape (1, 1)
            pi: tensor shape (1,) with probability of selected action
        """
        if not actionable_indices:
            raise ValueError("No actionable nodes provided to GraphPolicyNet.")

        node_feats = node_feats.to(self.device)
        edge_index = edge_index.to(self.device)

        x = self.node_embed(node_feats)

        for _ in range(self.mp_iterations):
            x = self.message_passing(x, edge_index)

        # Global mean pooling for value
        xg = x.mean(dim=0, keepdim=True)
        value = self.value_head(xg)  # (1, 1)

        if only_v:
            # Return a dummy index and prob; caller ignores them
            return (
                torch.tensor([actionable_indices[0]], device=self.device),
                value,
                torch.ones(1, device=self.device),
            )

        # Policy over actionable nodes
        logits_all = self.policy_head(x).flatten()  # (N,)
        idx_tensor = torch.tensor(
            actionable_indices, dtype=torch.long, device=self.device
        )
        logits = logits_all[idx_tensor]  # (A,)

        probs = torch.softmax(logits, dim=0)
        sel_local = segmented_sample(probs)  # scalar
        sel_idx = idx_tensor[sel_local]  # scalar global node index
        sel_prob = probs[sel_local]

        return sel_idx.unsqueeze(0), value, sel_prob.unsqueeze(0)

    def update(
        self,
        loss: torch.Tensor,
    ) -> float:
        self.opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.opt.step()
        return float(norm)

    def save(self, filepath: str) -> None:
        """
        Save model and optimizer state to a file.
        """
        torch.save(
            {
                "model_state": self.state_dict(),
                "optimizer_state": self.opt.state_dict(),
                "mp_iterations": self.mp_iterations,
                "lr": self.lr,
            },
            filepath,
        )

    def load(self, filepath: str, map_location: str | None = None) -> None:
        """
        Load model (and, if available, optimizer) state from a file.
        """
        checkpoint = torch.load(filepath, map_location=map_location or self.device)
        self.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            try:
                self.opt.load_state_dict(checkpoint["optimizer_state"])
            except Exception:
                # Optimizer state is optional; ignore incompatibilities.
                pass
