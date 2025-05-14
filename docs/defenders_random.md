# Defender Random

This agent implements a random strategy to block IP addresses in the environment. It starts with the same powers as a normal defender which means controlling all the hosts, and knowing all the hosts and networks and services and data. 

Then, it randomly selects a src host from the list of controlled hosts, then selects a target host from the list of controlled hosts, then it selects an IP from the list of known hosts, and it applies the action BlockIP(src host, target host, blocked_ip).


## False Positives
If any action done by any benign agent is blocked by the FW of the game, then it is counted as a false positive and the final reward of the defender is decreased a fixed amount per false positive.