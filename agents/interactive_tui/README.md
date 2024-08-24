# Interactive TUI agent

This is the main agent that should be used to play by humans. It can be played in several modes.

1. Human, without autocompletion of fields nor assistance.
2. Human, with autocompletion of fields, but without assistance.
3. Human, with autocompletion of fields and LLM assitance.

# Display
For reasons of the LLM prompt. The **known hosts** field is _not_ filled with the known hosts sent by the initial observation in the env. This is to avoid some errors in the LLM beliving that the hosts were also controlled.