"""
Sampling strategies for text generation
=========================================

Provides various sampling methods for controlling generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class SamplingStrategy(Enum):
    """Available sampling strategies."""

    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TYPICAL = "typical"
    BEAM_SEARCH = "beam_search"
    TEMPERATURE = "temperature"


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""

    strategy: SamplingStrategy = SamplingStrategy.TOP_P
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    typical_p: float = 0.95
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    max_length: int = 512
    seed: Optional[int] = None


class Sampler(ABC):
    """Abstract base class for samplers."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        if config.seed is not None:
            torch.manual_seed(config.seed)

    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next tokens from logits.

        Args:
            logits: Logits from model (batch_size, vocab_size)
            past_tokens: Previously generated tokens

        Returns:
            Tuple of (sampled_tokens, probabilities)
        """

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        past_tokens: torch.Tensor,
        penalty: float = 1.0,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0 or past_tokens is None:
            return logits

        # Gather logits for past tokens
        batch_size = logits.shape[0]
        for i in range(batch_size):
            for token_id in past_tokens[i]:
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty

        return logits


class GreedySampler(Sampler):
    """Greedy sampling - always pick highest probability token."""

    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy sampling."""
        # Apply repetition penalty
        logits = self.apply_repetition_penalty(logits, past_tokens, self.config.repetition_penalty)

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Select maximum probability tokens
        tokens = torch.argmax(probs, dim=-1)
        token_probs = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)

        return tokens, token_probs


class TopKSampler(Sampler):
    """Top-K sampling - sample from top K tokens."""

    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-K sampling."""
        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        # Apply repetition penalty
        logits = self.apply_repetition_penalty(logits, past_tokens, self.config.repetition_penalty)

        # Filter top-k
        k = min(self.config.top_k, logits.shape[-1])
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

        # Create filtered logits
        filtered_logits = torch.full_like(logits, -float("inf"))
        filtered_logits.scatter_(-1, top_k_indices, top_k_logits)

        # Sample from filtered distribution
        probs = F.softmax(filtered_logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        token_probs = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)

        return tokens, token_probs


class TopPSampler(Sampler):
    """Nucleus sampling - sample from smallest set with cumulative probability > p."""

    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-P (nucleus) sampling."""
        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        # Apply repetition penalty
        logits = self.apply_repetition_penalty(logits, past_tokens, self.config.repetition_penalty)

        # Sort logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Find cutoff
        cutoff_mask = cumulative_probs > self.config.top_p
        cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
        cutoff_mask[:, 0] = False

        # Set logits below cutoff to -inf
        sorted_logits[cutoff_mask] = -float("inf")

        # Restore original order
        filtered_logits = torch.zeros_like(logits)
        filtered_logits.scatter_(-1, sorted_indices, sorted_logits)

        # Sample from filtered distribution
        probs = F.softmax(filtered_logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        token_probs = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)

        return tokens, token_probs


class TypicalSampler(Sampler):
    """Typical sampling - sample tokens close to expected information content."""

    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Typical sampling."""
        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        # Apply repetition penalty
        logits = self.apply_repetition_penalty(logits, past_tokens, self.config.repetition_penalty)

        # Calculate entropy and typical values
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)

        # Calculate absolute difference from entropy
        log_probs = torch.log(probs + 1e-10)
        typical_values = torch.abs(log_probs + entropy)

        # Sort by typical values (lower is more typical)
        sorted_typical, sorted_indices = torch.sort(typical_values, dim=-1)
        sorted_probs = torch.gather(probs, -1, sorted_indices)

        # Find cutoff for typical_p
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = cumulative_probs > self.config.typical_p
        cutoff_mask[:, 1:] = cutoff_mask[:, :-1].clone()
        cutoff_mask[:, 0] = False

        # Create filtered distribution
        filtered_probs = sorted_probs.clone()
        filtered_probs[cutoff_mask] = 0
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        # Sample from filtered typical distribution
        sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
        tokens = torch.gather(sorted_indices, -1, sampled_idx).squeeze(-1)
        token_probs = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)

        return tokens, token_probs


class BeamSearchSampler(Sampler):
    """Beam search sampling for finding high-probability sequences."""

    def __init__(self, config: SamplingConfig):
        super().__init__(config)
        self.beam_scores = None
        self.beam_tokens = None
        self.finished_beams = []

    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search sampling.

        Note: This is a simplified version. Full beam search requires
        maintaining multiple beams throughout generation.
        """
        batch_size, vocab_size = logits.shape
        num_beams = self.config.num_beams

        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        # Apply repetition penalty
        logits = self.apply_repetition_penalty(logits, past_tokens, self.config.repetition_penalty)

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Initialize beams on first step
        if self.beam_scores is None:
            # Get top-k beams
            beam_scores, beam_tokens = torch.topk(log_probs[0], num_beams, dim=-1)
            self.beam_scores = beam_scores
            self.beam_tokens = beam_tokens.unsqueeze(0)

            # Return best beam token
            return beam_tokens[0].unsqueeze(0), torch.exp(beam_scores[0]).unsqueeze(0)

        # Expand beams
        expanded_scores = self.beam_scores.unsqueeze(-1) + log_probs
        flat_scores = expanded_scores.view(-1)

        # Get top scoring new beams
        top_scores, top_indices = torch.topk(flat_scores, num_beams, dim=-1)

        # Decode beam and token indices
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        # Update beam state
        self.beam_scores = top_scores
        self.beam_tokens = torch.cat([self.beam_tokens[:, beam_indices], token_indices.unsqueeze(0)], dim=0)

        # Return best beam's latest token
        best_token = token_indices[0]
        best_prob = torch.exp(top_scores[0] - self.beam_scores[0])

        return best_token.unsqueeze(0), best_prob.unsqueeze(0)

    def get_best_sequence(self) -> torch.Tensor:
        """Get the best sequence from beam search."""
        if self.beam_tokens is not None:
            return self.beam_tokens[:, 0]
        return torch.tensor([])


def create_sampler(config: SamplingConfig) -> Sampler:
    """Create a sampler based on configuration."""
    samplers = {
        SamplingStrategy.GREEDY: GreedySampler,
        SamplingStrategy.TOP_K: TopKSampler,
        SamplingStrategy.TOP_P: TopPSampler,
        SamplingStrategy.TYPICAL: TypicalSampler,
        SamplingStrategy.BEAM_SEARCH: BeamSearchSampler,
    }

    sampler_class = samplers.get(config.strategy, TopPSampler)
    return sampler_class(config)
