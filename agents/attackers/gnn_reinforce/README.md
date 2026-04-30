# GNN agents 

A single factored GNN policy trained end-to-end with REINFORCE directly against the
game server. The policy must learn the full attack chain
autonomously. See [`docs/blackbox_pure_gnn_arch.md`](docs/blackbox_pure_gnn_arch.md)
for the full architecture.

## Training
To train the GNN for 100,000 episodes:

```bash
python blackbox_pure_gnn_agent.py --episodes 100000 --wandb --wandb_project <your project> --wandb_entity <your entity> \
  --checkpoint_interval 10000 --eval_episodes 100 --port 9010 --eval_interval 1000 --eval_temperature 0.001 \
  --train_temperature_start 1.0 --curiosity_anneal_frac 0.1 --entropy_beta 0.1 --entropy_min 0.005 \
  --train_temperature_anneal_frac 0.7 --train_temperature_end 0.001
```

Notes:
- wandb is optional

## Evaluation

```bash
python blackbox_pure_gnn_agent.py --episodes 100 --curiosity_weight 0.0 --port 9011 --eval --eval_temperature 0.0 --weights <model weights>
```

`--verbose` can be used to log the actions that are taken by the agent

## Tests

```bash
python -m pytest tests/ -v
```

