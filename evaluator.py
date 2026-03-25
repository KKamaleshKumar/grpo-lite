"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

# --- Precompile regexes once (fast + consistent) ---
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
SOFT_RE = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)

# "Strict" but not insane: requires tags in order, allows any content, allows trailing whitespace
STRICT_RE = re.compile(
    r"^\s*<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s*$",
    re.DOTALL,
)

INT_RE = re.compile(r"[+-]?\d+$")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "gsm8k":
        return GSM8kEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")



class GSM8kEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    """
    
    def __init__(self):
        self.num_reward_functions = 5



    def _extract_xml_answer(self, text: str) -> str:
        """Extract answer between <answer>...</answer>. Returns '' if missing."""
        text = _normalize_newlines(text)
        m = ANSWER_RE.search(text)
        return m.group(1).strip() if m else ""
    
    
    def _parse_int(self, s: str) -> Optional[int]:
        s = s.strip()
        if INT_RE.fullmatch(s):
            try:
                return int(s)
            except ValueError:
                return None
        return None
    
    
    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """
        Reward for correct answer.
        Expects `answer` to be a list of ground-truth answers (strings or ints).
        If both parse as ints, compares ints; else compares stripped strings.
        """
        responses = [c[0]["content"] for c in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
    
        rewards: List[float] = []
        for pred, gt in zip(extracted, answer):
            if not pred:               # no tags at all
                rewards.append(-1.0)  # worse penalty than just wrong answer?
                continue

            gt_str = str(gt).strip()
            pred_str = pred.strip()
    
            pred_i = self._parse_int(pred_str)
            gt_i = self._parse_int(gt_str)
    
            if pred_i is not None and gt_i is not None:
                rewards.append(2.0 if pred_i == gt_i else -1.0)   #negative penalties for wrong answer
            else:
                rewards.append(2.0 if pred_str == gt_str else -1.0)   #negative penalties for wrong answer
    
        return rewards
    
    
    def _int_format_reward(self, completions) -> List[float]:
        """Reward if <answer> contains a single integer (allows +/-)."""
        responses = [c[0]["content"] for c in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if INT_RE.fullmatch(a.strip()) else 0.0 for a in extracted]
    
    
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward only if the entire response is strictly wrapped and newline-formatted."""
        responses = [_normalize_newlines(c[0]["content"]) for c in completions]
        return [0.5 if STRICT_RE.fullmatch(r) else 0.0 for r in responses]
    
    
    def _soft_format_reward(self, completions) -> List[float]:
        """
        Reward if tags appear in the right order anywhere in the response.
        Uses search (NOT match) and DOTALL so multiline content works.
        """
        responses = [_normalize_newlines(c[0]["content"]) for c in completions]
        return [0.5 if SOFT_RE.search(r) else 0.0 for r in responses]
    
    
    def _xml_count_reward(self, completions) -> List[float]:
        """
        Shaped reward for XML formatting (dense, less brittle).
        - Gives partial credit for having tags
        - Penalizes text before <reasoning> and after </answer>
        - Caps penalties so you don't nuke cold start
        """
        def score(text: str) -> float:
            text = _normalize_newlines(text)
    
            r_open = text.count("<reasoning>") ==1      #check if there is only one <reasoning> tag
            r_close = text.count("</reasoning>") ==1      #check if there is only one </reasoning> tag
            a_open = text.count("<answer>") ==1      #check if there is only one <answer> tag   
            a_close = text.count("</answer>") ==1      #check if there is only one </answer> tag    
    
            # base tag presence rewards
            s = 0.0
            s += 0.10 if r_open else 0.0
            s += 0.10 if r_close else 0.0
            s += 0.10 if a_open else 0.0
            s += 0.10 if a_close else 0.0
    
            # order bonus (only if tags exist)
            if r_open and r_close and a_open and a_close:
                if text.find("<reasoning>") < text.find("</reasoning>") < text.find("<answer>") < text.find("</answer>"):
                    s += 0.10  # up to 0.50 total possible from this function
    
            # targeted penalties (capped)
            # penalty for any text before <reasoning>
            if "<reasoning>" in text:
                pre = text.split("<reasoning>", 1)[0]
                s -= min(0.25, 0.001 * len(pre))
    
            # penalty for any text after </answer>
            if "</answer>" in text:
                post = text.split("</answer>", 1)[-1]
                s -= min(0.25, 0.001 * len(post))
    
            return s
    
        responses = [c[0]["content"] for c in completions]
        return [score(r) for r in responses]



    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._correctness_reward(prompts, completions, answer),
            self._int_format_reward(completions),
            self._strict_format_reward(completions),
            self._soft_format_reward(completions),
            self._xml_count_reward(completions)
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/int_reward_func": reward_per_func[1].item(), 
            "rewards/strict_format_reward_func": reward_per_func[2].item(),
            "rewards/soft_format_reward_func": reward_per_func[3].item(),
            "rewards/xmlcount_reward_func": reward_per_func[4].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'integer_format': reward_scores[1].item(),
            'strict_format': reward_scores[2].item(),
            'soft_format': reward_scores[3].item(),
            'xml_count': reward_scores[4].item()
        }