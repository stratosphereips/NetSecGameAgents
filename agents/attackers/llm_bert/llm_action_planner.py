"""
@file llm_action_planner.py

@brief Implementation of an LLM-based action planner with ModernBERT integration for reactive agent systems.

This script defines classes and methods to facilitate interaction with language models (LLMs),
manage configuration files, and parse responses from LLM queries. The core functionality includes
planning actions based on observations and memory, parsing LLM responses, and dynamically loading
configuration files using YAML. Stage 2 has been replaced with two ModernBERT calls for action
type classification and masked language modeling.

"""

import sys
import os
from os import path
import yaml
import logging
import json
from dotenv import dotenv_values
from openai import OpenAI
from tenacity import retry, stop_after_attempt
import jinja2
import torch
# Compatibility shim: some Torch builds on Python 3.12 don't support torch.compile (Dynamo).
# Transformers' ModernBERT uses @torch.compile; make it a no-op if unsupported to prevent import-time failure.
try:
    _orig_torch_compile = getattr(torch, "compile", None)
    if _orig_torch_compile is None:
        def _noop_compile(fn=None, **kwargs):
            return fn
        torch.compile = _noop_compile  # type: ignore[attr-defined]
    else:
        def _safe_compile(fn=None, **kwargs):
            try:
                return _orig_torch_compile(fn, **kwargs)
            except Exception:
                return fn
        torch.compile = _safe_compile  # type: ignore[attr-defined]
except Exception:
    pass
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
import ast
import gc

import re
from collections import Counter

# Import validate_responses - works when run as module or script
try:
    from . import validate_responses
except ImportError:
    import validate_responses

from netsecgame import ActionType, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state


class ConfigLoader:
    """Class to handle loading YAML configurations."""
    
    @staticmethod
    def load_config(file_name: str = 'prompts.yaml') -> dict:
        possible_paths = [
            path.join(path.dirname(__file__), file_name),
            path.join(path.dirname(path.dirname(__file__)), file_name),
            path.join(path.dirname(path.dirname(path.dirname(__file__))), file_name),
        ]
        for yaml_file in possible_paths:
            if path.exists(yaml_file):
                with open(yaml_file, 'r') as file:
                    return yaml.safe_load(file)
        raise FileNotFoundError(f"{file_name} not found in expected directories.")


ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}


class ModernBERTActionClassifier:
    """Handler for the first ModernBERT model that classifies action types."""
    
    def __init__(self, model_path: str):
        # Force CPU usage to avoid CUDA compatibility issues
        self.device = torch.device("cpu")
        # Ensure we can load local directories and not treat them as hub repo ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.label2id = {
            "ScanNetwork": 0,
            "ScanServices": 1,
            "ExploitService": 2,
            "FindData": 3,
            "ExfiltrateData": 4
        }
        
        self.id2label = {
            0: "ScanNetwork",
            1: "ScanServices", 
            2: "ExploitService",
            3: "FindData",
            4: "ExfiltrateData"
        }
    
    def predict_action_type(self, prompt: str) -> tuple:
        """Predict action type from the complete prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=8048,  
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()
        
        predicted_action = self.id2label[predicted_class_id]
        return predicted_action, confidence


class ModernBERTMaskedLM:
    """Handler for the second ModernBERT model for masked language modeling - EXACT training implementation."""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure we can load local directories and not treat them as hub repo ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize logger
        self.logger = logging.getLogger("ModernBERT-MLM")
        
        # Ensure special tokens are available - EXACT training setup
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            
        # Resize embeddings if tokenizer changed - EXACT training setup
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def fix_format_and_remove_spaces(self, text):
        """EXACT training preprocessing - Convert dict strings to JSON AND remove ALL spaces between tokens"""
        # First convert dict strings to JSON - EXACT training logic
        if isinstance(text, str) and text.startswith("{'"):
            try:
                dict_obj = ast.literal_eval(text)
                text = json.dumps(dict_obj)
            except Exception:
                pass
        
        # Convert to string
        text = str(text)
        
        # Remove spaces before all special tokens - EXACT training preprocessing
        text = text.replace(" [MASK]", "[MASK]")
        text = text.replace(" [PAD]", "[PAD]")
        
        # Remove spaces between special tokens - EXACT training preprocessing
        text = text.replace("[MASK] [PAD]", "[MASK][PAD]")
        text = text.replace("[PAD] [MASK]", "[PAD][MASK]")
        text = text.replace("[PAD] [PAD]", "[PAD][PAD]")
        text = text.replace("[MASK] [MASK]", "[MASK][MASK]")
        
        return text
    
    def predict_action_full(self, prompt: str, masked_action: str, max_length: int = 8192, iterations: int = 50) -> tuple:
        """
        Single-pass inference: predict all [MASK] tokens in one forward pass.
        """
        self.model.eval()

        # Convert dict string to JSON if needed - EXACT training logic
        if isinstance(masked_action, str) and masked_action.startswith("{'"):
            try:
                masked_action = json.dumps(ast.literal_eval(masked_action))
            except Exception:
                pass

        masked_action = self.fix_format_and_remove_spaces(masked_action)

        original_text = f"Action: {masked_action}\n\nPrompt: {prompt}"

        # Log the exact masked prompt fed into the MLM
        try:
            self.logger.info(f"MLM masked prompt (exact input):\n{original_text}")
        except Exception:
            pass
        predictions = []
        raw_predictions = []

        # Tokenize once and get predictions for all mask positions
        inputs = self.tokenizer(
            original_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False  # Training used padding=False for inference
        ).to(self.device)

        input_ids = inputs["input_ids"][0]
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) > 0:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]  # [seq_len, vocab]

            # Collect predictions for each mask position (left-to-right order)
            for pos in mask_positions.tolist():
                token_id = int(torch.argmax(logits[pos]).item())
                token_str_raw = self.tokenizer.decode([token_id])
                # Keep raw token for logging, but drop newlines before reinsertion
                token_str_clean = token_str_raw.replace("\n", "")
                predictions.append(token_str_clean)
                raw_predictions.append(token_str_raw)

            # Cleanup tensors
            del outputs, logits, inputs
            torch.cuda.empty_cache()
            gc.collect()

        filled_text = original_text
        for tok in predictions:
            filled_text = filled_text.replace("[MASK]", tok, 1)

        # Log the predicted tokens for traceability
        try:
            if raw_predictions:
                self.logger.info(f"MLM predicted tokens (raw): {raw_predictions}")
            self.logger.info(f"MLM predicted tokens (cleaned): {predictions}")
            self.logger.info(f"MLM filled text (raw, includes specials):\n{filled_text}")
        except Exception:
            pass

        # Extract result (strip prompt section)
        if "Action: " in filled_text:
            result = filled_text.split("Action: ", 1)[1].strip()
            if "\n\nPrompt: " in result:
                result = result.split("\n\nPrompt: ")[0].strip()
        else:
            result = filled_text

        # Keep raw for logging, but return a cleaned JSON string for parsing
        raw_result = result
        try:
            self.logger.info(f"MLM result before cleaning: {raw_result}")
        except Exception:
            pass

        cleaned_result = raw_result.replace("[PAD]", "").replace("[UNK]", "")
        return cleaned_result, predictions


class LLMActionPlanner:
    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None, 
                 config: dict = None, use_reasoning: bool = False, use_reflection: bool = False, 
                 use_self_consistency: bool = False, classifier_model_path: str = None, 
                 mlm_model_path: str = None):
        self.model = model_name
        self.config = config or ConfigLoader.load_config()
        self.use_reasoning = use_reasoning
        self.use_reflection = use_reflection
        self.use_self_consistency = use_self_consistency
        self.modernbert_prompts = []


        if "gpt" in self.model:
            env_config = dotenv_values(".env")
            api_key = env_config.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found. Please set it in a .env file or as an environment variable.")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI(base_url=api_url, api_key="ollama")

        self.memory_len = memory_len
        self.logger = logging.getLogger("REACT-agent")
        self.update_instructions(goal.lower())
        self.prompts = []
        self.states = []
        self.responses = []
        
        # Initialize ModernBERT models
        self.action_classifier = None
        self.masked_lm = None
        
        if classifier_model_path:
            resolved_classifier = classifier_model_path
            if not path.isabs(resolved_classifier):
                candidate = path.join(path.dirname(__file__), classifier_model_path)
                if path.isdir(candidate):
                    resolved_classifier = candidate
            self.logger.info(f"Loading action classifier from: {resolved_classifier}")
            self.action_classifier = ModernBERTActionClassifier(resolved_classifier)
            
        if mlm_model_path:
            resolved_mlm = mlm_model_path
            if not path.isabs(resolved_mlm):
                candidate = path.join(path.dirname(__file__), mlm_model_path)
                if path.isdir(candidate):
                    resolved_mlm = candidate
            self.logger.info(f"Loading masked language model from: {resolved_mlm}")
            self.masked_lm = ModernBERTMaskedLM(resolved_mlm)
        
        # EXACT training data format templates 
        self.action_templates = {
            "ScanNetwork": {
                "action": "ScanNetwork",
                "parameters": {
                    "target_network": "[MASK].[MASK].[MASK].[MASK]/[MASK]",
                    "source_host": "[MASK].[MASK].[MASK].[MASK]"
                }
            },
            "ScanServices": {
                "action": "ScanServices", 
                "parameters": {
                    "target_host": "[MASK].[MASK].[MASK].[MASK]",
                    "source_host": "[MASK].[MASK].[MASK].[MASK]"
                }
            },
            "ExploitService": {
                "action": "ExploitService",
                "parameters": {
                    "target_host": "[MASK].[MASK].[MASK].[MASK]",
                    "target_service": "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]",
                    "source_host": "[MASK].[MASK].[MASK].[MASK]"
                }
            },
            "FindData": {
                "action": "FindData",
                "parameters": {
                    "target_host": "[MASK].[MASK].[MASK].[MASK]",
                    "source_host": "[MASK].[MASK].[MASK].[MASK]"
                }
            },
            "ExfiltrateData": {
                "action": "ExfiltrateData",
                "parameters": {
                    "target_host": "[MASK].[MASK].[MASK].[MASK]",
                    "data": {
                        "owner": "[MASK] [MASK] [MASK]",
                        "id": "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"
                    },
                    "source_host": "[MASK].[MASK].[MASK].[MASK]"
                }
            }
        }

    def get_prompts(self) -> list:
        return self.prompts

    def get_responses(self) -> list:
        return self.responses
    
    def get_states(self) -> list:
        return self.states
    
    def get_modernbert_prompts(self) -> list:
        return self.modernbert_prompts

    
    def update_instructions(self, new_goal: str) -> None:
        template = jinja2.Environment().from_string(self.config['prompts']['INSTRUCTIONS_TEMPLATE'])
        self.instructions = template.render(goal=new_goal)

    def create_mem_prompt(self, memory_list: list) -> str:
        prompt = ""
        for memory, goodness in memory_list:
            prompt += f"You have taken action {memory} in the past. This action was {goodness}.\n"
        return prompt

    @retry(stop=stop_after_attempt(3))
    def openai_query(self, msg_list: list, max_tokens: int = 60, model: str = None, fmt=None, temperature: float = 0.0):
        llm_response = self.client.chat.completions.create(
            model=model or self.model,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=fmt or {"type": "text"},
        )
        return llm_response.choices[0].message.content, llm_response.usage

    def parse_response(self, llm_response: str, state: Observation.state):
        """EXACT training parsing logic - no fallbacks or cleaning"""
        response_dict = {"action": None, "parameters": None}
        valid = False
        action = None

        try:
            # Direct JSON parsing 
            response = json.loads(llm_response)
            action_str = response.get("action", None)
            action_params = response.get("parameters", None)
            
            if action_str and action_params:
                valid, action = create_action_from_response(response, state)
                response_dict["action"] = action_str
                response_dict["parameters"] = action_params
                # Ensure we only report valid if we produced a concrete Action object
                if valid and action is None:
                    self.logger.warning("Validated response did not produce an actionable object; marking invalid.")
                    valid = False
            else:
                self.logger.warning("Missing action or parameters in response.")
                
        except json.JSONDecodeError:
            self.logger.error("Failed to parse response as JSON.")
            self.logger.error(f"Raw response: {llm_response}")
            # No fallback - just mark as invalid
            response_dict["action"] = "InvalidJSON"
            response_dict["parameters"] = {"raw_response": llm_response}
        except KeyError:
            self.logger.error("Missing keys in response.")
        
        return valid, response_dict, action

    def remove_reasoning(self, text):
        match = re.search(r'</think>(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def check_repetition(self, memory_list):
        repetitions = 0
        past_memories = []
        for memory, goodness in memory_list:
            if memory in past_memories:
                repetitions += 1
            past_memories.append(memory)
        return repetitions

    def get_self_consistent_response(self, messages, temp=0.4, max_tokens=1024, n=3):
        candidates = []
        total_tokens_used = 0
        for _ in range(n):
            response, usage = self.openai_query(messages, temperature=temp, max_tokens=max_tokens)
            total_tokens_used += usage.total_tokens
            candidates.append(response.strip())

        counts = Counter(candidates)
        most_common = counts.most_common(1)
        if most_common:
            self.logger.info(f"Self-consistency candidates: {counts}")
            return most_common[0][0], total_tokens_used
        return candidates[0], total_tokens_used

    def build_complete_training_prompt(self, stage1_response: str, status_prompt: str, memory_prompt: str) -> str:
        """
        Build the complete prompt that EXACTLY matches the training data format.
        """
        # Get the prompt components from config
        cot_prompt = self.config['prompts']['COT_PROMPT']
        q4 = self.config['questions'][3]['text']
        
        complete_prompt = f"{self.instructions}\n\n"
        complete_prompt += f"Current status:\n{status_prompt}\n\n"
        complete_prompt += f"{memory_prompt}\n\n" if memory_prompt.strip() else ""
        complete_prompt += f"{cot_prompt}\n\n"
        complete_prompt += f"{stage1_response}\n\n"
        complete_prompt += f"{q4}"
        
        return complete_prompt

    def get_action_with_modernbert(self, stage1_response: str, observation: Observation, status_prompt: str, memory_prompt: str) -> tuple:
        """
        Replace Stage 2 with ModernBERT calls using EXACT training implementation.
        """
        if not self.action_classifier or not self.masked_lm:
            self.logger.error("ModernBERT models not initialized.")
            return False, {"action": "ModelError", "parameters": {"error": "ModernBERT models not loaded"}}, None, 0
        
        try:
            complete_prompt = self.build_complete_training_prompt(stage1_response, status_prompt, memory_prompt)

            self.modernbert_prompts.append(complete_prompt)  

            
            self.logger.info(f"Complete prompt for ModernBERT:\n{complete_prompt[:500]}...")
            
            self.logger.info(f"Classifying action type with complete prompt...")
            predicted_action, confidence = self.action_classifier.predict_action_type(complete_prompt)
            self.logger.info(f"Predicted action: {predicted_action} (confidence: {confidence:.4f})")
            
            if predicted_action not in self.action_templates:
                self.logger.error(f"Unknown action type: {predicted_action}")
                return False, {"action": "UnknownAction", "parameters": {"predicted": predicted_action}}, None, 0
            
            masked_template = self.action_templates[predicted_action]
            masked_action_str = json.dumps(masked_template)
            
            self.logger.info(f"Using masked template: {masked_action_str}")
            
            filled_action, predictions = self.masked_lm.predict_action_full(
                prompt=complete_prompt,
                masked_action=masked_action_str,
                max_length=8192,  # Match training max_length
                iterations=50
            )
            
            self.logger.info(f"MLM predictions: {predictions}")
            self.logger.info(f"Filled action: {filled_action}")
            
            is_valid, response_dict, action = self.parse_response(filled_action, observation.state)
            
            self.logger.info(f"Final parsed action - Valid: {is_valid}, Response: {response_dict}")
            
            return is_valid, response_dict, action, 0  # No LLM tokens used in this stage
            
        except Exception as e:
            self.logger.error(f"Error in ModernBERT action generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, {"action": "BERTError", "parameters": {"error": str(e)}}, None, 0

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list) -> tuple:
        self.states.append(observation.state.as_json())
        status_prompt = create_status_from_state(observation.state)
        q1 = self.config['questions'][0]['text']
        memory_prompt = self.create_mem_prompt(memory_buf)
        total_tokens_used = 0

        repetitions = self.check_repetition(memory_buf)
        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": q1},
        ]
        self.logger.info(f"Text sent to the LLM: {messages}")

        # Stage 1: Get reasoning from LLM
        if self.use_self_consistency:
            response, tokens = self.get_self_consistent_response(messages, temp=repetitions/9, max_tokens=1024)
            total_tokens_used += tokens
        else:
            response, usage = self.openai_query(messages, max_tokens=1024)
            total_tokens_used += usage.total_tokens

        if self.use_reflection:
            reflection_prompt = [
                {
                    "role": "user",
                    "content": f"""
                    Instructions: {self.instructions}
                    Task: {q1}

                    Status: {status_prompt}
                    Memory: {memory_prompt}

                    Reasoning:
                    {response}

                    Is this reasoning valid given the Instructions, Status, and Memory?
                    - If YES, repeat it exactly.
                    - If NO, output the corrected reasoning only (no commentary).
                    """
                }
            ]
            response, usage = self.openai_query(reflection_prompt, max_tokens=1024)
            total_tokens_used += usage.total_tokens

        if self.use_reasoning:
            response = self.remove_reasoning(response)
        self.logger.info(f"(Stage 1) Response from LLM: {response}")

        # Stage 2: Use ModernBERT with EXACT training implementation
        self.logger.info("Using ModernBERT for Stage 2 action generation with EXACT training implementation...")
        is_valid, response_dict, action, bert_tokens = self.get_action_with_modernbert(
            response, observation, status_prompt, memory_prompt
        )
        total_tokens_used += bert_tokens  # Should be 0 for BERT models
        
        # Format response for consistency with original code
        final_response = json.dumps(response_dict, indent=2)
        self.responses.append(final_response)
        self.logger.info(f"(Stage 2) ModernBERT Response: {final_response}")
        print(f"(Stage 2) ModernBERT Response: {final_response}")
        
        return is_valid, response_dict, action, total_tokens_used
