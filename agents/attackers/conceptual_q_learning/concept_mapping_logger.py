# Authors: Claude AI Assistant
# Enhanced logging system for conceptual Q-learning agent
# Provides detailed tracing of action-concept mappings

import json
import logging
from typing import Dict, Any, Set, List
from datetime import datetime
from AIDojoCoordinator.game_components import Action, ActionType, GameState, Observation


class ConceptMappingLogger:
    """
    Enhanced logger for tracking concept-to-IP mappings and action conversions
    in the conceptual Q-learning agent
    """
    
    def __init__(self, logger: logging.Logger, verbose: bool = True):
        self.logger = logger
        self.verbose = verbose
        self.episode_number = 0
        self.step_number = 0
        
    def set_episode_step(self, episode: int, step: int):
        """Update current episode and step numbers for context"""
        self.episode_number = episode
        self.step_number = step
    
    def log_concept_conversion_start(self, observation: Observation):
        """Log the start of IP-to-concept conversion"""
        if not self.verbose:
            return
            
        state = observation.state
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"CONCEPT CONVERSION START - Episode {self.episode_number}, Step {self.step_number}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"INPUT STATE (Real IPs):")
        self.logger.info(f"  Known Networks: {list(state.known_networks)}")
        self.logger.info(f"  Known Hosts: {list(state.known_hosts)}")
        self.logger.info(f"  Controlled Hosts: {list(state.controlled_hosts)}")
        self.logger.info(f"  Known Services: {dict(state.known_services)}")
        self.logger.info(f"  Known Data: {dict(state.known_data)}")
        self.logger.info(f"  Known Blocks: {dict(state.known_blocks)}")
    
    def log_concept_mapping_table(self, concept_mapping: Dict[str, Dict], ip_to_concept: Dict):
        """Log the complete concept mapping table"""
        if not self.verbose:
            return
            
        self.logger.info(f"\nCONCEPT MAPPING TABLE:")
        self.logger.info(f"  IP to Concept Mapping: {ip_to_concept}")
        for category, mapping in concept_mapping.items():
            if mapping:  # Only log non-empty mappings
                self.logger.info(f"  {category}:")
                for concept, real_value in mapping.items():
                    self.logger.info(f"    {concept} -> {real_value}")
    
    def log_concept_conversion_complete(self, concept_observation: Any, concept_mapping: Dict):
        """Log the completion of IP-to-concept conversion"""
        if not self.verbose:
            return
            
        state = concept_observation.observation.state
        self.logger.info(f"\nCONCEPT CONVERSION COMPLETE:")
        self.logger.info(f"  OUTPUT STATE (Concepts):")
        self.logger.info(f"    Known Networks: {list(state.known_networks)}")
        self.logger.info(f"    Known Hosts: {list(state.known_hosts)}")
        self.logger.info(f"    Controlled Hosts: {list(state.controlled_hosts)}")
        self.logger.info(f"    Known Services: {dict(state.known_services)}")
        self.logger.info(f"    Known Data: {dict(state.known_data)}")
        self.logger.info(f"{'='*80}\n")
    
    def log_action_conversion_start(self, concept_action: Action, concept_observation: Any):
        """Log the start of concept-to-action conversion"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"ACTION CONVERSION START - Episode {self.episode_number}, Step {self.step_number}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"CONCEPT ACTION: {concept_action}")
        self.logger.info(f"  Type: {concept_action.type}")
        self.logger.info(f"  Parameters: {concept_action.parameters}")
        
        if hasattr(concept_observation, 'concept_mapping'):
            self.logger.info(f"  Available Concept Mappings:")
            for category, mapping in concept_observation.concept_mapping.items():
                if mapping:
                    self.logger.info(f"    {category}: {mapping}")
    
    def log_parameter_conversion(self, param_name: str, concept_value: Any, real_value: Any, conversion_method: str):
        """Log individual parameter conversions"""
        if not self.verbose:
            return
            
        self.logger.info(f"  PARAMETER CONVERSION:")
        self.logger.info(f"    {param_name}: {concept_value} -> {real_value}")
        self.logger.info(f"    Method: {conversion_method}")
    
    def log_action_conversion_complete(self, real_action: Action):
        """Log the completion of concept-to-action conversion"""
        self.logger.info(f"\nACTION CONVERSION COMPLETE:")
        self.logger.info(f"  REAL ACTION: {real_action}")
        self.logger.info(f"  Type: {real_action.type}")
        self.logger.info(f"  Parameters: {real_action.parameters}")
        self.logger.info(f"{'='*80}\n")
    
    def log_action_history_update(self, concept_action: Action, action_history: Set):
        """Log when an action is added to history"""
        self.logger.info(f"ACTION HISTORY UPDATE:")
        self.logger.info(f"  Added to history: {concept_action}")
        self.logger.info(f"  Total actions in history: {len(action_history)}")
        if self.verbose:
            self.logger.info(f"  Current history: {list(action_history)}")
    
    def log_valid_actions_generation(self, valid_actions: List[Action], action_history: Set):
        """Log the generation of valid concept actions"""
        self.logger.info(f"\nVALID ACTIONS GENERATION:")
        self.logger.info(f"  Generated {len(valid_actions)} valid actions")
        self.logger.info(f"  Actions in history (excluded): {len(action_history)}")
        
        if self.verbose and valid_actions:
            self.logger.info(f"  Valid actions:")
            for i, action in enumerate(valid_actions):
                self.logger.info(f"    {i+1}: {action}")
    
    def log_q_value_update(self, state_id: int, concept_action: Action, old_q: float, new_q: float, 
                          reward: float, max_next_q: float, alpha: float, gamma: float):
        """Log Q-value updates with full calculation details"""
        self.logger.info(f"\nQ-VALUE UPDATE:")
        self.logger.info(f"  State ID: {state_id}")
        self.logger.info(f"  Action: {concept_action}")
        self.logger.info(f"  Old Q-value: {old_q:.4f}")
        self.logger.info(f"  New Q-value: {new_q:.4f}")
        self.logger.info(f"  Calculation: {old_q:.4f} + {alpha} * ({reward} + {gamma} * {max_next_q:.4f} - {old_q:.4f}) = {new_q:.4f}")
    
    def log_episode_summary(self, episode: int, steps: int, reward: float, 
                           q_table_size: int, epsilon: float, end_reason: str):
        """Log episode completion summary"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EPISODE {episode} SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  Steps: {steps}")
        self.logger.info(f"  Final Reward: {reward}")
        self.logger.info(f"  Q-table Size: {q_table_size}")
        self.logger.info(f"  Current Epsilon: {epsilon:.4f}")
        self.logger.info(f"  End Reason: {end_reason}")
        self.logger.info(f"{'='*60}\n")
    
    def log_error(self, message: str, exception: Exception = None):
        """Log error messages with optional exception details"""
        self.logger.error(f"CONCEPT MAPPING ERROR: {message}")
        if exception:
            self.logger.error(f"  Exception: {str(exception)}")
            self.logger.error(f"  Type: {type(exception).__name__}")
    
    def create_mapping_visualization(self, concept_mapping: Dict, ip_to_concept: Dict) -> str:
        """Create a visual representation of the concept mapping"""
        viz_lines = ["CONCEPT MAPPING VISUALIZATION:", ""]
        
        # Create reverse mapping for better visualization
        concept_to_ips = {}
        for ip, concept in ip_to_concept.items():
            if concept not in concept_to_ips:
                concept_to_ips[concept] = []
            concept_to_ips[concept].append(str(ip))
        
        for concept, ips in concept_to_ips.items():
            viz_lines.append(f"  {concept}")
            for ip in ips:
                viz_lines.append(f"    └── {ip}")
            viz_lines.append("")
        
        return "\n".join(viz_lines)