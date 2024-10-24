# Defender Random

This agent implements a random strategy to block IP addresses in the environment. It starts with the same powers as a normal defender which means controlling all the hosts, and knowing all the hosts and networks and services and data. 

Then, it randomly selects a src host from the list of controlled hosts, then selects a target host from the list of controlled hosts, then it selects an IP from the list of known hosts, and it applies the action BlockIP(src host, target host, blocked_ip).