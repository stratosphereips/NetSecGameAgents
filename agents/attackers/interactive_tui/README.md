# Interactive TUI agent

This is the main agent that should be used to play by humans. It can be played in several modes.

1. Human, without autocompletion of fields nor assistance.
2. Human, with autocompletion of fields, but without assistance.
3. Human, with autocompletion of fields and LLM assitance.

# Display
For reasons of the LLM prompt. The **known hosts** field is _not_ filled with the known hosts sent by the initial observation in the env. This is to avoid some errors in the LLM beliving that the hosts were also controlled.

## Installation
To install the TUI agent, follow the installation guide in the NetSecGameAgents with `[tui]` option:

```
pip install -e .[tui]
```
It is recommended to install the agent in a virtual environment.

## Runnig the agent
This agent can be run in two ways: CLI and web based. I CLI, there is the normal and guided mode. The guided mode only allows to select action parameters from the GameState while in the normal mode, the user has to type the parameters and there are no restrictions. The web-based interface only supports the guided mode.

To use the CLI run:
```
python -m  interactive_tui --mode=<normal,guided>
```

For web-based interface
```
textual serve interactive_tui --port=<PORT>