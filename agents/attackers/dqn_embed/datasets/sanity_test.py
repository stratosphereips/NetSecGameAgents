from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

import torch
from sentence_transformers.evaluation import SentenceEvaluator
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

model_name = "Qwen/Qwen3-Embedding-0.6B"
model_name = "stratosphere/Qwen3-Embedding-0.6B-netsecgame-finetuned"
model = SentenceTransformer(model_name)
# dataset = load_dataset("stratosphere/netsecgame-embedding-finetuning")

import torch
from sentence_transformers import util

def run_sensitivity_test(model, baseline_json):
    # 1. IP Mutation (Same structure, different IPs)
    # We just shift the last octet of every IP in the string
    ip_mutated = baseline_json.replace(".2", ".99").replace(".3", ".88")
    
    # 2. Structural Mutation - Minor (Add one host)
    # This simulates a small but real state change
    minor_mutation = baseline_json.replace(
        '"known_hosts": [', 
        '"known_hosts": [{"ip": "10.99.99.99"}, '
    )
    
    # 3. Structural Mutation - Major (Empty the networks)
    major_mutation = baseline_json.replace('"known_data": {', '"known_data": {"192.168.2.5": ["id": "sensitive_info"]}')

    # Encode all scenarios
    states = [baseline_json, ip_mutated, minor_mutation, major_mutation]
    embeddings = model.encode(states, convert_to_tensor=True)
    
    # Calculate Cosine Similarities against the Baseline
    scores = util.cos_sim(embeddings[0], embeddings[1:])[0]
    
    print("--- Sensitivity Analysis ---")
    print(f"1. IP Change Only:      {scores[0]:.4f} (Goal: > 0.98)")
    print(f"2. Minor Structural:    {scores[1]:.4f} (Goal: < 0.90)")
    print(f"3. Major Structural:    {scores[2]:.4f} (Goal: < 0.70)")
    
    # Success Logic
    gap = scores[0] - scores[1]
    if gap > 0.05:
        print(f"\nSUCCESS: The model is {gap*100:.1f}% more sensitive to structure than IP noise.")
    else:
        print("\nWARNING: Model still struggles to distinguish structure from data.")

# Usage:

test_state = '{"known_networks": [{"ip": "192.168.3.0", "mask": 24}, {"ip": "192.168.2.0", "mask": 24}, {"ip": "213.47.23.192", "mask": 26}, {"ip": "192.168.1.0", "mask": 24}], "known_hosts": [{"ip": "192.168.2.3"}, {"ip": "213.47.23.195"}, {"ip": "192.168.1.4"}, {"ip": "192.168.1.2"}, {"ip": "192.168.1.3"}], "controlled_hosts": [{"ip": "213.47.23.195"}, {"ip": "192.168.2.3"}], "known_services": {"192.168.2.3": [{"name": "3389/tcp, ms-wbt-server", "type": "passive", "version": "10.0.19041", "is_local": false}, {"name": "powershell", "type": "passive", "version": "10.0.19041", "is_local": true}]}, "known_data": {}, "known_blocks": {}}'
run_sensitivity_test(model, test_state)