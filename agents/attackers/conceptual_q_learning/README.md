# Conceptual Attacker Agent

The conceptual attacker agent is a modification to the Q-learning attacker to avoid depending on IP addresses to play the game, and instead convert each IP address into a concept, just as humans do when they attack a network.

# Install
Install the dependencies of this agent with 

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e ..
python -m pip install -e ".[conceptual_q_learning]"
```

# Run the Agent
If the NetSecGame server is running in localhost, port 9000/TCP, then:

```
python -m agents.attackers.conceptual_q_learning.conceptual_q_agent --host localhost --port 9000 --episodes 1 --experiment_id test-1 --env_conf ../AIDojoCoordinator/netsecenv_conf.yaml
```

Testing and evaluation trajectories are recorded by default in the
`TrajectoryRecorder` JSONL format. Select the output directory with:

```bash
--trajectoriesdir /path/to/trajectories
```

Use `--no-record-trajectories` to disable trajectory recording.

## Action-generation ablations

The conceptual action generator supports independent, opt-in ablations. With
none of the following flags, the agent uses the original action-generation
behavior. The selected values are stored in the W&B run configuration under
`action_generation_options`.

### Per-family conceptual-filter ablations

| Flag | Effect |
| --- | --- |
| `--no_filter_scan_network` | Keep `ScanNetwork` available, but generate every controlled-source and known-network combination instead of restricting scans to internal concepts. |
| `--no_filter_find_services` | Keep `FindServices` available, but do not apply internal-host or already-known-services filtering. |
| `--no_filter_exploit_service` | Keep `ExploitService` available, but do not filter external concepts, local services, controlled targets, or self-targets. |
| `--no_filter_find_data` | Keep `FindData` available, but do not filter external concepts or hosts whose data is already known. |
| `--no_filter_exfiltrate_data` | Keep `ExfiltrateData` available, but do not filter external data sources, logfiles, or data already present at the destination. |

These flags never remove the named action family. They remove the conceptual
knowledge supplied specifically to that family, leaving the Q-learning agent
to learn which generated actions are useful.

History filtering, source branching, and firewall filtering remain independent.
For example, `--no_filter_scan_network` still suppresses a repeated scan unless
you also use `--allow_repeated_network_scans` or
`--allow_repeated_actions`.

### Rule ablations

| Flag | Effect |
| --- | --- |
| `--allow_repeated_actions` | Ignore conceptual action history for every action family. |
| `--single_source` | Use the first deterministic internal controlled host as source for scanning, service discovery, exploitation, and `FindData`. Exfiltration sources remain hosts containing data. |
| `--allow_repeated_network_scans` | Ignore conceptual action history for `ScanNetwork` only. |
| `--allow_service_rescans` | Generate `FindServices` for hosts whose services are already known. |
| `--include_local_services` | Generate exploits for services marked as local. |
| `--allow_exploit_controlled_hosts` | Generate exploits against already controlled hosts. |
| `--allow_find_data_rescans` | Generate `FindData` for hosts whose data is already known. |
| `--prohibit_find_data_self_targeting` | Require different source and target hosts for `FindData`. The baseline permits self-targeting. |
| `--include_logfile_exfiltration` | Generate exfiltration actions for data with id `logfile`. |
| `--allow_duplicate_data_exfiltration` | Generate exfiltration when the destination already contains the same data id. |
| `--exfiltrate_to_external_only` | Restrict exfiltration destinations to controlled concepts containing `external`, normally the C2 host. |
| `--ignore_firewall` | Do not prune actions using the agent's known firewall blocks. |

Hyphenated aliases are also accepted, for example
`--allow-repeated-actions`.

Run one ablation by adding its flag to the normal command:

```bash
python -m agents.attackers.conceptual_q_learning.conceptual_q_agent \
  --host localhost \
  --port 9011 \
  --episodes 1000 \
  --experiment_id no-history-1 \
  --allow_repeated_actions
```

For controlled comparisons, change only the ablation flag and keep the agent
seed, coordinator seed, environment configuration, training budget, and
evaluation schedule identical.
