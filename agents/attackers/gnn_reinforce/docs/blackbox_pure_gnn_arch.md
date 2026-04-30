# Blackbox Pure GNN Agent — Architecture

`blackbox_pure_gnn_agent.py` implements a **factored GNN policy** trained end-to-end
with REINFORCE directly against the NetSecGame server. It has no goal conditioning (no
LLM, no symbolic monitor) — the policy must discover and execute the full attack chain
autonomously.

---

## 1. Motivation: Why Factor the Action Space?

Every valid NetSecGame action is a tuple of parameters like
`(type, source_host, target_host, target_service)`. A naive policy could score all
valid actions jointly, but the branching factor is
`O(|types| × |hosts| × |targets| × ...)`, which grows quickly and makes credit
assignment hard — the policy can't tell which part of the action was good or bad.

The factored policy decomposes each decision into **up to four sequential
sub-decisions**, each conditioned on the choices made before it:

```
Step 1:  What kind of action?           (5-way choice:  scan, find services, ...)
Step 2:  From which host?               (pick one of the controlled hosts)
Step 3:  At what target?                (pick a host, network, or data object)
Step 4:  Which specific sub-target?     (only for ExfiltrateData)
```

This reduces the effective branching factor from `O(|valid_actions|)` to
`O(5) + O(|hosts|) + O(|targets|) + O(|sub-targets|)` and means each head only has
to solve a much smaller classification problem. It also lets each head attend to the
right part of the graph embedding.

---

## 2. State Representation: Heterogeneous Graph

The game state (which hosts are known, which services run where, what data has been
found) is converted into a **graph** — a data structure where entities are **nodes**
and relationships between them are **edges**. This is done by `state_to_pyg()`
(from `policy_netsec.py`), which produces a PyTorch Geometric `HeteroData` object.

The graph is **heterogeneous**: nodes come in different types and edges come in
different types, each representing a different kind of relationship. This lets the
neural network treat hosts, services, and data differently while still reasoning
about their connections.

### 2.1 Node types

| Type      | Contents                                     | Feature vector |
|-----------|----------------------------------------------|----------------|
| `network` | known IP networks (CIDR blocks)              | `[scan_attempts_norm, has_yielded_hosts, known_host_count_norm, 0, …]` (16-dim) |
| `host`    | known + controlled hosts                     | `[is_controlled, is_public, findservices_attempts_norm, finddata_attempts_norm, exploit_attempts_norm, 0, …]` (16-dim) |
| `service` | known services (deduplicated by equality)    | `[port_norm, is_local, 0, …]` (16-dim) |
| `data`    | known data objects (deduplicated)            | `[is_exfiltrated, origin_is_public, exfil_attempts_norm, 0, …]` (16-dim) |

Each node carries a 16-dimensional feature vector. Host nodes encode whether the agent
controls the host, whether it has a public IP, and three normalised per-action
attempt counters for the current episode — how many times the policy has tried
`FindServices`, `FindData`, and `ExploitService` against this host. Each counter is
clipped at 10 and scaled to `[0, 1]`, so a host that has been repeatedly targeted by
a given action is explicitly distinguishable from one that has not. Network nodes
encode a normalised `ScanNetwork` attempt count plus two structural signals — a
boolean `has_yielded_hosts` (true once ≥1 known host lies inside the network's CIDR)
and a normalised count of known hosts in that CIDR — so the policy can tell which
networks are still opaque versus productively scanned. Service nodes encode a
normalised port and `is_local` flag. Data nodes encode whether the data is already
exfiltrated (present on a controlled public host), whether its origin host is
public, and a normalised `ExfiltrateData` attempt count for that specific data item.
Raw IP addresses are deliberately excluded so the policy generalises across IP
randomisation without retraining.

**Service deduplication caveat.** If the same service (e.g. `ssh/tcp/8.0`) runs on
multiple hosts, `state_to_pyg` creates only **one** service node — whichever host
sorts first claims the edge. This means the single `ssh` node's learned embedding
reflects only one host's neighbourhood, even though ssh is available elsewhere.
This is a known limitation. In the current blackbox agent, `ExploitService` is treated
as a host-level decision and the service is filled deterministically after the host is
chosen, which avoids learning over a nuisance exploit-service choice.

### 2.2 Edge types

Edges represent relationships between nodes. Each edge is directed (has a source and
target node):

```
host  ──[in]──────────► network       "this host belongs to that network"
network ──[contains]──► host          "that network contains this host" (reverse)
host  ──[runs]─────────► service      "this host runs that service"
host  ──[stores]───────► data         "this host stores that data"
service ──[rev_runs]──► host          "reverse edge so host can aggregate service context"
data ──[rev_stores]──► host           "reverse edge so host can aggregate data context"
```

The `in`/`contains`, `runs`/`rev_runs`, and `stores`/`rev_stores` pairs are
bidirectional (both directions stored explicitly) so information can flow both ways
during message passing (section 3.3).

Node ordering within each type is always deterministic (sorted by `str()` key) to
prevent graph topology from changing arbitrarily between calls.

### 2.3 Index mappings

`state_to_pyg` returns two auxiliary dicts alongside the graph, and it optionally
accepts an `attempt_counts: AttemptCounts` dataclass from the agent so the graph can
encode per-episode, per-action interaction history (ScanNetwork on networks;
FindServices/FindData/ExploitService on hosts; ExfiltrateData on data):

- `object_to_idx[type][game_object] → int` — maps a game object (e.g. an `IP`) to its
  row index in the node feature matrix.
- `idx_to_object[type][int] → game_object` — the inverse.

These are needed to translate between the policy's index-space decisions and the
game's object-space actions.

---

## 3. Network Architecture: `FactoredGNNPolicy`

The neural network takes the graph from section 2 and produces an action. At a high
level it does two things: (1) it runs a GNN over the graph so every node gets a
learned embedding vector that captures its role in the network topology, and (2) it
uses those embeddings in up to four sequential decision heads to pick the action
step-by-step: action type → source host → primary target → secondary target.

Each "head" is a small MLP (multi-layer perceptron — just a standard
fully-connected neural network) that outputs a score. Scores are turned into
probabilities via softmax, and the action component is sampled from that
probability distribution. Temperature controls the sharpness: high during early
training for exploration, low during evaluation for near-greedy behaviour.

### 3.1 Overview

The architecture has two parts: a **shared GNN backbone** that reads the graph and
produces per-node embeddings, and **decision heads** that run sequentially to select
an action.

```
                     GameState ─────────────────────────────────┐
                         │                                      │
                    state_to_pyg()                              │
                         │                                      │
                   HeteroData graph                             │
                         │                                      │
               ┌─────────▼──────────┐                           │
               │   Node Encoders    │  one linear per node type │
  SHARED       └─────────┬──────────┘                           │
  GNN                    │                                      │
  BACKBONE     ┌─────────▼──────────┐                           │
               │  2 × GATv2 Layers  │  message passing (§3.3)   │
               └─────────┬──────────┘                           │
                         │  per-node embeddings h_v ∈ R^H       │
               ┌─────────▼──────────┐      ┌────────────────────▼──────┐
               │   Global Pool      │      │  State Summary σ          │
               │   ⇒ g ∈ R^H        │      │  3-dim phase-of-attack    │
               └─────────┬──────────┘      │  counts, computed         │
                         │                 │  directly from GameState  │
                         │                 │  (NOT from the backbone)  │
                         │                 └─────────────┬─────────────┘
                         │                               │
                         └────────────────┐   ┌──────────┘
                                          ▼   ▼
  SEQUENTIAL   ┌────────────────────────────────────────────────────┐
  DECISION     │  Step 1: type_head(g ‖ σ)     → action type  τ*    │
  HEADS        │            │                                       │
  (each uses   │  Step 2: src_head(h_v, g, τ*) → source host s*     │
  g + all      │            │                                       │
  previous     │  Step 3: tgt_head(h_v, g, τ*, s*)   → target t*    │
  choices)     │            │                                       │
               │  Step 4: sec_head(h_v, g, τ*, s*, t*) → u*         │
               │          (ExfiltrateData only)                     │
               └────────────────────────────────────────────────────┘
                                         │
                              Action(τ*, s*, t*, u*)
```

The four heads are applied **sequentially**, not in parallel — each receives the
outputs of all previous heads as additional input. This is what makes the factored
log-probability decomposition valid (section 3.10).

`sec_head` is active only for `ExfiltrateData` (picks the destination host after the
data is chosen). `ExploitService` is treated as a host-level decision in this agent:
once the host is chosen, the service is filled deterministically with a canonical
valid service for that host. For all other action types `lp_sec = 0`, `H_sec = 0`.

### 3.2 Node Encoders

Each node type τ has its own linear encoder (no shared weights):

$$h_v^{(0)} = \text{ReLU}(W_\tau\, x_v), \quad W_\tau \in \mathbb{R}^{H \times 16}$$

This allows the model to learn type-specific feature projections before message
passing begins.

### 3.3 GNN Backbone: Message Passing with Attention

The core idea of a Graph Neural Network is **message passing**: each node collects
information from its neighbours, transforms it, and uses it to update its own
representation. After two rounds of this, each node's embedding captures not just its
own features but also the structure of its two-hop neighbourhood — e.g. a host "knows"
what services it runs and what network it belongs to.

This architecture uses **GATv2** (Graph Attention Network v2), which adds a learnable
**attention mechanism**: when a node collects messages from its neighbours, it assigns
different importance weights to each neighbour rather than treating them equally. The
weight is computed from both the sender's and receiver's current embeddings:

$$e_{uv} = \mathbf{a}^\top \text{LeakyReLU}\!\left(W_s h_u + W_t h_v + b\right)$$

$$\alpha_{uv} = \frac{\exp(e_{uv})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{kv})}$$

$$h_v^{(\ell)} = \text{ReLU}\!\left(\sum_{u \in \mathcal{N}(v)} \alpha_{uv}\, W_{\text{msg}}\, h_u^{(\ell-1)}\right)$$

In plain terms: $e_{uv}$ is a raw "relevance score" between neighbours $u$ and $v$.
These scores are normalised via softmax to get attention weights $\alpha_{uv}$ that
sum to 1. Node $v$'s new embedding is a weighted sum of transformed neighbour
embeddings, passed through ReLU.

Because the graph is heterogeneous (different edge types), a separate GATv2 layer runs
per edge type (e.g. `host→service` and `host→network` use different weight matrices).
`HeteroConv` with `aggr='sum'` merges the results when a node receives messages from
multiple edge types.

The backbone stacks **two** such layers (`num_gnn_layers=2`). Only edges with
`numel() > 0` are passed into each layer, so sparse early-game states (few edges) do
not cause errors.

### 3.4 Global Embedding

After the two GATv2 layers, a single graph-level vector is computed:

$$g = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{1}{|V_\tau|} \sum_{v \in V_\tau} h_v^{(L)}$$

i.e. mean-pool within each node type, then mean across node types. This gives a
fixed-size summary of the entire network state regardless of topology size.

### 3.5 Decision Table

| Action type      | Step 3 (tgt_head)  | Step 4 (sec_head)           |
|-----------------|--------------------|-----------------------------|
| `ScanNetwork`   | `target_network`   | —                           |
| `FindServices`  | `target_host`      | —                           |
| `FindData`      | `target_host`      | —                           |
| `ExploitService`| `target_host`      | — (service filled deterministically) |
| `ExfiltrateData`| `data`             | `target_host` (destination) |

For `ExploitService`, the host is chosen by the policy and the service is then filled
deterministically from the valid actions for that host. The current implementation
uses a canonical representative: the lexicographically smallest valid service for the
chosen `(source_host, target_host)` pair. This removes a nuisance decision when the
environment's exploit outcome does not depend on which valid service is selected.

For `ExfiltrateData`, the data item is chosen first (conditioned on the already-chosen
source host), then the destination is scored. This matters for the win condition:
exfiltration must reach a public host, and the destination head learns to prefer it.

### 3.6 Action Type Head

A two-layer MLP maps the concatenation of the global embedding and a **state summary
vector** $\sigma \in \mathbb{R}^3$ to 5 logits (one per action type). The state
summary is computed directly from the raw `GameState` — it is *not* derived from
the GNN backbone or the mean-pooled embedding. It provides direct phase-of-attack
awareness that is otherwise lost in the mean-pooled global embedding:

| Feature | Meaning | Signals |
|---------|---------|---------|
| $\sigma_0$ | Controlled hosts with no known data | Need `FindData` |
| $\sigma_1$ | Known data not yet exfiltrated | Need `ExfiltrateData` |
| $\sigma_2$ | Uncontrolled hosts with known services | Can `ExploitService` |

All values are clipped at 10 and scaled to $[0, 1]$. Without these features, the
type head sees only $g$, which barely changes when a single host flips to controlled
— making the type head unable to distinguish "just exploited, need FindData" from
"should exploit next."

$$z_{\text{type}} = \text{MLP}_{\text{type}}([g \;\|\; \sigma]) \in \mathbb{R}^5$$

$$z_{\text{type}}[i] \leftarrow -10^9 \quad \text{if action type } i \notin \mathcal{A}_{\text{valid}}$$

$$\pi_{\text{type}}(\cdot \mid s) = \text{softmax}(z_{\text{type}} / \tau_e)$$

$$\tau^* \sim \pi_{\text{type}}$$

The chosen type index is looked up in a learned `Embedding` table to produce a
type context vector $t \in \mathbb{R}^H$, which conditions all subsequent heads
(source, target, and secondary).

### 3.7 Source Host Head

All five action types carry a `source_host` parameter. For each candidate source host
$v_s$ (those appearing in valid actions of type $\tau^*$), the head scores the
concatenation of the host's node embedding, the global embedding, and the type
context:

$$\text{score}(v_s) = \text{MLP}_{\text{src}}\!\left([h_{v_s}^{(L)} \;\|\; g \;\|\; t\,]\right) \in \mathbb{R}$$

$$\pi_{\text{src}}(\cdot \mid s, \tau^*) = \text{softmax}([\text{score}(v_s)]_{v_s \in \mathcal{S}_{\text{valid}}})$$

$$s^* \sim \pi_{\text{src}}$$

The source host embedding $h_{s^*}^{(L)}$ is then carried forward to condition the
target head.

### 3.8 Target Entity Head (Primary)

The primary target type depends on the chosen action type:

| Action type      | Step 3 target param | Step 3 node type |
|-----------------|---------------------|------------------|
| `ScanNetwork`   | `target_network`    | `network`        |
| `FindServices`  | `target_host`       | `host`           |
| `ExploitService`| `target_host`       | `host`           |
| `FindData`      | `target_host`       | `host`           |
| `ExfiltrateData`| `data`              | `data`           |

For each candidate target node $v_t$ of the correct type, the head scores a
four-part concatenation:

$$\text{score}(v_t) = \text{MLP}_{\text{tgt}}\!\left([h_{v_t}^{(L)} \;\|\; g \;\|\; t \;\|\; h_{s^*}^{(L)}]\right) \in \mathbb{R}$$

$$\pi_{\text{tgt}}(\cdot \mid s, \tau^*, s^*) = \text{softmax}([\text{score}(v_t)]_{v_t \in \mathcal{T}_{\text{valid}}})$$

$$t^* \sim \pi_{\text{tgt}}$$

The embedding of $t^*$ is saved as $h_{t^*}^{(L)}$ and passed to the secondary head
when step 4 is active. Candidates are then filtered to `a.parameters[tgt_param] == t*`
before step 4.

### 3.9 Secondary Head (ExfiltrateData only)

| Action type      | Step 4 param       | Step 4 node type |
|-----------------|---------------------|------------------|
| `ExfiltrateData`| `target_host` (dest)| `host`           |

For `ExfiltrateData`, after the data item $t^*$ has been chosen, the candidates are
restricted to valid destination hosts for that data. The head scores a five-part
concatenation:

$$\text{score}(v_u) = \text{MLP}_{\text{sec}}\!\left([h_{v_u}^{(L)} \;\|\; g \;\|\; t \;\|\; h_{s^*}^{(L)} \;\|\; h_{t^*}^{(L)}]\right) \in \mathbb{R}$$

$$\pi_{\text{sec}}(\cdot \mid s, \tau^*, s^*, t^*) = \text{softmax}([\text{score}(v_u)]_{v_u \in \mathcal{U}_{\text{valid}}})$$

$$u^* \sim \pi_{\text{sec}}$$

For all other action types `sec_head` is not called; $\log \pi_{\text{sec}} = 0$ and
$H_{\text{sec}} = 0$. In particular, `ExploitService` contributes only type, source,
and target-host terms to the joint log-probability.

### 3.10 Combined Log-probability and Entropy

The four sub-decisions are made sequentially, each conditioned on all previous
choices, so the joint log-probability is an exact sum:

$$\log \pi(a \mid s) = \log \pi_{\text{type}} + \log \pi_{\text{src}} + \log \pi_{\text{tgt}} + \log \pi_{\text{sec}}$$

where $\log \pi_{\text{sec}} = 0$ for the four action types that do not use step 4.

The policy entropy used for regularisation is the sum of the four sub-distribution
entropies:

$$H[\pi(\cdot \mid s)] = H[\pi_{\text{type}}] + H[\pi_{\text{src}}] + H[\pi_{\text{tgt}}] + H[\pi_{\text{sec}}]$$

This is an upper bound on the true joint entropy (which would require marginalising
over the earlier choices), but it is cheap to compute and sufficient as a
regularisation signal.

---

## 4. Training: REINFORCE with Entropy and Curiosity

### 4.1 Return computation and value baseline

Discounted returns are computed in reverse:

$$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k, \quad \gamma = 0.99$$

A learned value baseline $V_\phi(s_t)$ (computed by `value_head` from the global graph
embedding concatenated with the state summary $\sigma$) is subtracted to form
advantages:

$$A_t = G_t - V_\phi(s_t)$$

Advantages are **not** normalised per-episode. Per-episode normalisation forces every
episode to have zero-mean advantages, which means half the actions in a winning episode
get penalised and half the actions in a losing episode get reinforced — corrupting
credit assignment in the factored action space. The value baseline already approximately
centres the advantages; additional normalisation is harmful.

### 4.2 REINFORCE loss

$$\mathcal{L} = \underbrace{-\sum_{t} \log \pi(a_t \mid s_t) \cdot A_t}_{\text{policy loss}} \;+\; \underbrace{0.5 \cdot \text{MSE}(V_\phi, G)}_{\text{value loss}} \;-\; \underbrace{\beta_e \cdot \frac{1}{T}\sum_t H[\pi(\cdot \mid s_t)]}_{\text{entropy bonus}}$$

The value loss trains the baseline; the entropy term keeps the policy exploratory early
in training.

### 4.3 Entropy annealing

The entropy coefficient decays geometrically over the course of training from
$\beta_0$ to $\beta_{\min}$:

$$\beta_e = \beta_0 \cdot \left(\frac{\beta_{\min}}{\beta_0}\right)^{e / E}$$

where $e$ is the current episode and $E$ is the total episode budget. With defaults
$\beta_0 = 0.1$, $\beta_{\min} = 0.01$, this is a one-decade decay — the policy
starts highly exploratory and gradually commits, but retains meaningful stochasticity
throughout training thanks to the raised floor.

### 4.4 Curiosity bonus

To encourage exploration of new hosts, services, and data in the early phase of
training, a bonus is added to the environment reward at each step:

$$r_t' = r_t + c_e \cdot \left(\mathbf{1}[\Delta\text{hosts}>0] + \mathbf{1}[\Delta\text{services}>0] + \mathbf{1}[\Delta\text{data}>0]\right)$$

The curiosity weight $c_e$ decays linearly to zero over `curiosity_anneal_frac`
(default 50%) of total training episodes:

$$c_e = \begin{cases} c_0 \cdot \left(1 - \dfrac{e}{f \cdot E}\right) & e < f \cdot E \\[6pt] 0 & e \geq f \cdot E \end{cases}$$

where $f$ is `curiosity_anneal_frac`. This keeps intrinsic motivation active through
the first half of training, preventing premature convergence to a local optimum that
only partially solves the attack chain.

### 4.5 Temperature annealing

All four decision heads divide their logits by a temperature $\tau_e$ before softmax.
During training, this temperature anneals linearly from a configurable start value
`train_temperature_start` toward the evaluation temperature `eval_temperature` over a
configurable fraction `train_temperature_anneal_frac` of the full run, then remains at
the evaluation temperature for the rest of training.

Let $\tau_{\text{start}}$ be `train_temperature_start`, $\tau_{\text{eval}}$ be
`eval_temperature`, $f_\tau$ be `train_temperature_anneal_frac`, and
$E_\tau = \max(1, \lfloor f_\tau \cdot E \rfloor)$. Then:

$$p_e = \min\!\left(\frac{e}{E_\tau}, 1\right)$$

$$\tau_e = \tau_{\text{start}} + (\tau_{\text{eval}} - \tau_{\text{start}}) \cdot p_e$$

This closes the distribution shift between training and evaluation: early episodes
explore broadly at high temperature, while late episodes train at the exact temperature
used during evaluation. By controlling both the start value and the anneal fraction,
the schedule can spend more or less of training in the near-greedy regime. The policy
is forced to learn action compositions that work under low-temperature sampling, rather
than relying on exploration-driven luck that doesn't transfer to low-temperature
evaluation.

### 4.6 Optimiser, learning rate schedule, and checkpointing

Adam is used with **ReduceLROnPlateau** scheduling: the learning rate (default
$10^{-4}$) is halved when the stochastic eval win rate stagnates for 20 consecutive
eval cycles, with a floor of $10^{-6}$. This preserves learning capacity as long as
the policy is improving and only reduces LR when the signal-to-noise ratio genuinely
requires it. Gradients are clipped to a max norm of 1.0.

After every `eval_interval` training episodes the policy is frozen (`eval()` mode,
low-temperature sampling with `temperature = eval_temperature`, `no_grad`) and run for
`eval_episodes` episodes against the live server. Win rate, average steps, and average
reward are logged to Weights & Biases. The checkpoint with the highest eval win rate is
saved to `--weights`.

---

## 5. End-to-end Data Flow (Single Step)

```
Observation (from server)
        │
  filter_log_files_from_state()     ← strip verbose logfile data entries
        │
  generate_valid_actions(state)     ← enumerate legal actions
        │
  canonicalize ExploitService       ← keep one canonical service per (source, target host)
        │
  state_to_pyg(state, attempt_counts)
                                   ← build HeteroData + index maps, inject per-action attempt history
        │
  Inference path
    ├── Stochastic eval / training:
    │     FactoredGNNPolicy.forward(...)
    │       ├── _gnn_forward()                        ← encode nodes, run 2× GATv2
    │       ├── _global_emb()                         ← mean-pool across all node types
    │       ├── type_head(g)                          ← 5-way masked softmax → τ*
    │       ├── src_head(h_src ‖ g ‖ t)              ← per-source-host score → s*
    │       ├── tgt_head(h_tgt ‖ g ‖ t ‖ h_s*)      ← per-target score → t*
    │       └── sec_head(h_sec ‖ g ‖ t ‖ h_s* ‖ h_t*)  ← ExfiltrateData only → u*
        │
  Reconstruct Action(τ*, s*, t*, u*)   ← u* only for ExfiltrateData; exploit service already canonicalized
        │
  agent.make_step(action)           ← send to server, receive next Observation
        │
  Store (log_prob, entropy, reward)
        │  [repeat until episode ends]
        │
  compute_returns(rewards)
  REINFORCE update (Adam step)
```

---

## 6. Key Design Notes

**No goal conditioning.** Unlike `agent_netsec.py` / `train_netsec.py`, there is no
`NodeGoal` or LLM layer. The policy must learn the full attack sequence (scan →
exploit → exfiltrate) as a single monolithic behaviour.

**Black-box environment.** `BlackBoxGNNAgent` communicates with the game server over
TCP using only the standard `BaseAgent` API (`register`, `make_step`,
`request_game_reset`). It never reads internal server state — only the `Observation`
returned per step.

**IP-agnostic features.** Raw IPs are never put into node features. Hosts use
`[is_controlled, is_public, findservices_attempts_norm, finddata_attempts_norm,
exploit_attempts_norm, 0, …]`; networks and data nodes carry their own attempt
counters plus structural signals (see §2.1). All history features depend only on
the agent's own interaction counts within the current episode, not on any fixed
identifier, so the trained weights still transfer across episodes with different IP
assignments without retraining.

**Deterministic node ordering.** `state_to_pyg` sorts all node sets by their string
representation before assigning indices. This is critical: Python `set` iteration
order is not stable across calls, so without sorting the same graph would map to
different index orderings each time, making the GNN output non-reproducible.

**Log-probability decomposition validity.** The factored log-prob sum
`lp_type + lp_src + lp_tgt + lp_sec` is the true log-probability of the chosen action
only when the sub-decisions are made sequentially and each distribution is conditioned
on all previous choices — which is exactly how `forward()` is structured. For three-step
action types `lp_sec = 0`, so the sum reduces to three terms.

**Service deduplication limitation still affects graph context.** Because `state_to_pyg`
creates only one graph node per unique service (regardless of how many hosts run it),
the GNN embedding for that service reflects only the first host's neighbourhood.
Canonicalizing `ExploitService` removes the need to learn over service identity in this
agent, which avoids the most direct exploit-time failure mode. However, service
deduplication still means the graph itself loses some host-specific service context. A
future fix would make service nodes per-host in the graph.

**Structurally identical hosts are only partially disambiguated.** The per-action
attempt counters break some symmetry between otherwise identical hosts by telling
the policy which hosts have already been targeted by `FindServices`, `FindData`, or
`ExploitService` in the current episode. However, hosts with the same
`is_controlled` / `is_public` flags, the same graph neighbourhood, and the same
attempt-count vector will still have identical inputs. The policy therefore remains
partially blind in highly symmetric states; the counters help avoid redundant-action
loops, but they do not solve the general symmetry problem.

---

## 7. Potential Improvements

The following extensions would improve the policy's ability to distinguish structurally
similar nodes without sacrificing IP-agnosticity (raw IPs remain excluded from features).

**Success/failure attempt counters.** The current counters track *attempts* only —
they increment whether or not the action succeeded. Splitting each into success and
failure streams would let the policy distinguish "already tried and failed" from
"already tried and succeeded" without re-querying the environment. This preserves
the lightweight per-episode bookkeeping approach.

**Graph positional encodings.** Assign each node a positional feature that encodes its
structural role in the graph. Two standard approaches: (1) **random features** — sample
a fixed-length random vector per node at episode start, so identical nodes get different
IDs throughout the episode; (2) **Laplacian eigenvectors** — compute the top-$k$
eigenvectors of the graph Laplacian and use them as node features. Both are well-studied
techniques for breaking symmetry in GNNs (see Dwivedi et al., "Benchmarking Graph
Neural Networks", 2020).

**Recurrent memory (GRU/LSTM over timesteps).** Replace the current stateless
per-step policy with a recurrent architecture that maintains a hidden state across
timesteps. This lets the policy remember which hosts it already attempted and what
outcomes it observed, enabling it to avoid redundant actions (e.g. re-scanning a host
that yielded nothing). The GNN embeddings at each step would be fed through a GRU cell
whose hidden state carries forward temporal context.

**Per-host service nodes.** Replace the current deduplicated service graph (one node per
unique service) with per-host service nodes — if `ssh` runs on three hosts, create three
separate `ssh` nodes, each connected only to its own host. This would restore
host-specific service context in the graph and remove the residual information loss from
service deduplication.

---

## 8. Related Work (references to cross-check when writing up)

Reading list grouped by the architectural idea each supports. This is the long list
compiled from literature search on 2026-04-19; the paper's final related-work section
will be a filtered subset. Entries marked **[in paper]** are already cited in
`paper/main.tex`.

### 8.1 GATv2 and heterogeneous GNN backbones (§3.3)

- **[in paper]** Brody, Alon, Yahav. *How Attentive are Graph Attention Networks?*
  ICLR 2022. — GATv2 attention, fixes static-attention failure mode of GATv1.
- Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling. *Modeling Relational
  Data with Graph Convolutional Networks.* ESWC 2018. — R-GCN, canonical
  relation-typed GCN and precursor to `HeteroConv`.
- Hu, Dong, Wang, Sun. *Heterogeneous Graph Transformer.* WWW 2020. — Modern
  heterogeneous attention; useful if we want to motivate per-edge-type projections.

### 8.2 Factored / autoregressive action heads (§3.5–§3.10)

- **[in paper]** Vinyals et al. *Grandmaster level in StarCraft II using
  multi-agent RL.* Nature 2019. — AlphaStar's autoregressive entity-scoring head;
  closest large-scale precedent for type → source → target → secondary factorisation.
- **[in paper]** Tavakoli, Pardo, Kormushev. *Action Branching Architectures for
  Deep RL.* AAAI 2018. — BDQ; contrast point for *parallel* branching vs. our
  *sequential* (autoregressive) factorisation.
- Berner et al. *Dota 2 with Large Scale Deep RL.* arXiv:1912.06680, 2019. —
  OpenAI Five's factored action head with sub-action categoricals.
- Sharma, Suresh, Ramesh, Ravindran. *Learning to Factor Policies and Action-Value
  Functions.* arXiv:1705.07269, 2017. — Early factored-action-space paradigm;
  explicitly compares independent vs. sequential factorisation.
- Metz, Ibarz, Jaitly, Davidson. *Discrete Sequential Prediction of Continuous
  Actions for Deep RL.* arXiv:1705.05035, 2017. — Autoregressive decoding over
  action dimensions.

### 8.3 Pointer / attention scoring over variable entity sets (§3.7–§3.9)

- Vinyals, Fortunato, Jaitly. *Pointer Networks.* NeurIPS 2015. — Direct ancestor
  of variable-size attention-based action scoring over entity sets.
- **[in paper]** Huang, Ontañón. *A Closer Look at Invalid Action Masking in
  Policy Gradient Algorithms.* FLAIRS 2022. — Justifies masking logits to
  `-1e9` rather than resampling.

### 8.4 Relational / entity-attention RL (§2, §3)

- Zambaldi et al. *Relational Deep Reinforcement Learning* / *Deep RL with
  Relational Inductive Biases.* arXiv:1806.01830, 2018; ICLR 2019. — Foundational
  entity-attention policy, StarCraft II mini-games.
- Battaglia et al. *Relational inductive biases, deep learning, and graph
  networks.* arXiv:1806.01261, 2018. — Foundational framework for justifying
  graph-structured policies in RL.

### 8.5 GNN-structured policies in RL (§3 as a whole)

- Wang, Liu, Mohan, Zemel. *NerveNet: Learning Structured Policy with Graph
  Neural Networks.* ICLR 2018. — GNN policy with per-entity action heads;
  closest non-cyber structural analogue.
- Bapst et al. *Structured Agents for Physical Construction.* ICML 2019. —
  Object-centric GNN + RL with "pick an entity, then act on it" decomposition.
- Khalil, Dai, Zhang, Dilkina, Song. *Learning Combinatorial Optimization
  Algorithms over Graphs.* NeurIPS 2017. — GNN + RL where actions are node
  choices — the general pattern the ExploitService head instantiates.

### 8.6 GNN + RL for autonomous cyber defence (most directly parallel prior work)

This is the group reviewers will compare against first.

- Symes Thompson et al. *An Attentive Graph Agent for Topology-Adaptive Cyber
  Defence.* arXiv:2501.14700, 2025. — GAT on CybORG; closest architectural sibling.
- Nyberg, Johnson. *Structural Generalization in Autonomous Cyber Incident
  Response with Message-Passing Neural Networks and RL.* arXiv:2407.05775, 2024.
  — MPNN + PPO on CybORG, same topology-generalisation motivation as ours.
- Collyer et al. *Automated Cyber Defense with Generalizable Graph-based RL
  Agents.* arXiv:2509.16151, 2025. — Graph-based agents for CAGE challenges.
- *Towards a Generalisable Cyber Defence Agent for Real-World Computer Networks.*
  arXiv:2511.09114, 2025. — Most recent CAGE-style generalisation study.
- Standen et al. *CybORG: A Gym for the Development of Autonomous Cyber Agents.*
  IJCAI-21 AICS workshop. — Baseline problem-formulation citation for the
  cyber-RL line of work.
- Kunz et al. / Microsoft. *CyberBattleSim* technical report. — Microsoft's
  simulator; red-team counterpart to our offensive-agent setup.

### 8.7 Symmetry-breaking and positional encodings (§7 future work)

- Dwivedi, Joshi, Laurent, Bengio, Bresson. *Benchmarking Graph Neural Networks.*
  arXiv:2003.00982, 2020. — Already referenced inline; supports Laplacian
  eigenvector positional encodings.
- Murphy, Srinivasan, Rao, Ribeiro. *Relational Pooling for Graph Representations.*
  ICML 2019. — Theoretical grounding for random node features as symmetry breakers.

### 8.8 Curated aggregators

- Kim Hammar. *awesome-rl-for-cybersecurity* (GitHub).
  <https://github.com/Kim-Hammar/awesome-rl-for-cybersecurity> — Curated list;
  useful as a snowball-citation seed before final submission.
