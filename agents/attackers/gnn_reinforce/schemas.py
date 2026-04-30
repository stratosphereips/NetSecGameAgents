from typing import Literal
from pydantic import BaseModel, Field


class NodeGoal(BaseModel):
    """
    Structured JSON output schema for the LLM Strategic Layer to define a sub-goal
    grounded in a specific graph node.

    target_node_key is '{node_type}_{sorted_index}', matching the deterministic sort
    order used by state_to_pyg() in policy_netsec.py.
    """
    intent: Literal["discovery", "exploitation", "locate", "exfiltration"] = Field(
        ...,
        description="The high-level intent or operational goal. YOU MUST USE EXACTLY ONE OF THESE."
    )
    target_node_type: Literal["network", "host", "service", "data"] = Field(
        ...,
        description="The type of graph node the agent should target."
    )
    target_node_key: str = Field(
        ...,
        description=(
            "The specific node key in the format '{node_type}_{sorted_index}', "
            "e.g. 'host_0', 'network_1', 'data_2'. Must match the node keys listed "
            "in the state summary."
        )
    )
