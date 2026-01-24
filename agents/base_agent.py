import warnings
from netsecgame.agents.base_agent import BaseAgent

warnings.warn(
    "Importing BaseAgent from 'NetSecGameAgents.agents.base_agent' is deprecated. "
    "Please import directly from 'netsecgame' as follows: 'from netsecgame import BaseAgent'.",
    DeprecationWarning,
    stacklevel=2
)