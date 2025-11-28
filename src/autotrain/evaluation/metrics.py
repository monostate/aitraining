"""
Evaluation Metrics for AutoTrain Advanced
==========================================

Collection of metrics for model evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import torch


# Optional imports with fallback
try:
    from sacrebleu import corpus_bleu

    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score

    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score

    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Metric(ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Union[float, Dict[str, float]]:
        """Compute metric value."""

    def reset(self):
        """Reset internal state if any."""


class PerplexityMetric(Metric):
    """Perplexity metric for language modeling."""

    def compute(self, losses: Union[List[float], torch.Tensor], **kwargs) -> float:
        """
        Compute perplexity from losses.

        Args:
            losses: List of loss values or loss tensor

        Returns:
            Perplexity value
        """
        if isinstance(losses, torch.Tensor):
            avg_loss = losses.mean().item()
        else:
            avg_loss = np.mean(losses)

        perplexity = np.exp(avg_loss) if avg_loss < 100 else float("inf")
        return perplexity


class BLEUMetric(Metric):
    """BLEU score for text generation."""

    def __init__(self, n_gram: int = 4, smooth: bool = True):
        self.n_gram = n_gram
        self.smooth = smooth

    def compute(self, predictions: List[str], references: List[Union[str, List[str]]], **kwargs) -> Dict[str, float]:
        """
        Compute BLEU score.

        Args:
            predictions: Generated texts
            references: Reference texts (can be multiple per prediction)

        Returns:
            Dictionary with BLEU scores
        """
        if not SACREBLEU_AVAILABLE:
            # Fallback to simple implementation
            return self._simple_bleu(predictions, references)

        # Ensure references is list of lists
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Compute BLEU
        bleu = corpus_bleu(predictions, references)

        return {
            "bleu": bleu.score / 100.0,  # Normalize to 0-1
            "bleu_1": bleu.precisions[0] / 100.0 if len(bleu.precisions) > 0 else 0.0,
            "bleu_2": bleu.precisions[1] / 100.0 if len(bleu.precisions) > 1 else 0.0,
            "bleu_3": bleu.precisions[2] / 100.0 if len(bleu.precisions) > 2 else 0.0,
            "bleu_4": bleu.precisions[3] / 100.0 if len(bleu.precisions) > 3 else 0.0,
        }

    def _simple_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple BLEU implementation as fallback."""
        # Very basic n-gram overlap
        scores = []
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]

            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            # Count matching n-grams
            matches = 0
            for i in range(len(pred_tokens)):
                if i < len(ref_tokens) and pred_tokens[i] == ref_tokens[i]:
                    matches += 1

            score = matches / max(len(pred_tokens), 1)
            scores.append(score)

        return {"bleu": np.mean(scores)}


class ROUGEMetric(Metric):
    """ROUGE scores for text generation."""

    def __init__(self, rouge_types: List[str] = None):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL"]

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary with ROUGE scores
        """
        if not ROUGE_AVAILABLE:
            # Fallback to simple implementation
            return self._simple_rouge(predictions, references)

        scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        scores = {rouge_type: [] for rouge_type in self.rouge_types}

        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]

            result = scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                scores[rouge_type].append(result[rouge_type].fmeasure)

        # Average scores
        avg_scores = {}
        for rouge_type in self.rouge_types:
            avg_scores[rouge_type] = np.mean(scores[rouge_type])

        return avg_scores

    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple ROUGE implementation as fallback."""
        scores = []
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]

            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens) if pred_tokens else 0
            recall = overlap / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            scores.append(f1)

        return {"rouge1": np.mean(scores)}


class BERTScoreMetric(Metric):
    """BERTScore for semantic similarity."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "auto"):
        self.model_name = model_name

        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Compute BERTScore.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary with BERTScore values
        """
        if not BERTSCORE_AVAILABLE:
            return {"bert_score_f1": 0.0}

        # Compute BERTScore
        P, R, F1 = bert_score(predictions, references, model_type=self.model_name, device=self.device, verbose=False)

        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }


class AccuracyMetric(Metric):
    """Accuracy metric for classification."""

    def compute(
        self, predictions: Union[List[int], np.ndarray], references: Union[List[int], np.ndarray], **kwargs
    ) -> float:
        """
        Compute accuracy.

        Args:
            predictions: Predicted labels
            references: True labels

        Returns:
            Accuracy value
        """
        return accuracy_score(references, predictions)


class F1Metric(Metric):
    """F1 score for classification."""

    def __init__(self, average: str = "macro"):
        self.average = average

    def compute(
        self, predictions: Union[List[int], np.ndarray], references: Union[List[int], np.ndarray], **kwargs
    ) -> Dict[str, float]:
        """
        Compute F1 score.

        Args:
            predictions: Predicted labels
            references: True labels

        Returns:
            Dictionary with F1, precision, recall
        """
        f1 = f1_score(references, predictions, average=self.average, zero_division=0)
        precision = precision_score(references, predictions, average=self.average, zero_division=0)
        recall = recall_score(references, predictions, average=self.average, zero_division=0)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }


class ExactMatchMetric(Metric):
    """Exact match metric for QA and generation."""

    def __init__(self, normalize: bool = True, ignore_case: bool = True):
        self.normalize = normalize
        self.ignore_case = ignore_case

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute exact match score.

        Args:
            predictions: Predicted texts
            references: Reference texts

        Returns:
            Exact match score
        """
        matches = []
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]

            if self.normalize:
                pred = self._normalize_text(pred)
                ref = self._normalize_text(ref)

            if self.ignore_case:
                pred = pred.lower()
                ref = ref.lower()

            matches.append(float(pred == ref))

        return np.mean(matches)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re

        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()


class METEORMetric(Metric):
    """METEOR score for text generation."""

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute METEOR score.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            METEOR score
        """
        if not METEOR_AVAILABLE:
            # Fallback to simple overlap
            scores = []
            for pred, ref in zip(predictions, references):
                if isinstance(ref, list):
                    ref = ref[0]

                pred_tokens = set(pred.split())
                ref_tokens = set(ref.split())

                if not pred_tokens or not ref_tokens:
                    scores.append(0.0)
                    continue

                overlap = len(pred_tokens & ref_tokens)
                score = overlap / max(len(pred_tokens), len(ref_tokens))
                scores.append(score)

            return np.mean(scores)

        # Download required NLTK data
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except:
            pass

        scores = []
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]

            try:
                score = meteor_score([ref], pred)
                scores.append(score)
            except:
                scores.append(0.0)

        return np.mean(scores)


class CustomMetric(Metric):
    """Custom metric with user-defined function."""

    def __init__(self, metric_fn: callable, name: str = "custom"):
        """
        Initialize custom metric.

        Args:
            metric_fn: Function that takes (predictions, references) and returns score
            name: Name of the metric
        """
        self.metric_fn = metric_fn
        self.name = name

    def compute(self, predictions: List[Any], references: List[Any], **kwargs) -> Union[float, Dict[str, float]]:
        """Compute custom metric."""
        return self.metric_fn(predictions, references, **kwargs)


class MetricCollection:
    """Collection of metrics to compute together."""

    def __init__(self, metrics: Dict[str, Metric]):
        """
        Initialize metric collection.

        Args:
            metrics: Dictionary of metric name to Metric instance
        """
        self.metrics = metrics

    def compute_all(
        self, predictions: List[Any], references: List[Any], **kwargs
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute all metrics.

        Args:
            predictions: Model predictions
            references: Reference values

        Returns:
            Dictionary of metric results
        """
        results = {}
        for name, metric in self.metrics.items():
            try:
                result = metric.compute(predictions, references, **kwargs)
                if isinstance(result, dict):
                    # Flatten nested metrics
                    for sub_name, value in result.items():
                        results[f"{name}_{sub_name}" if sub_name != name else name] = value
                else:
                    results[name] = result
            except Exception as e:
                print(f"Failed to compute {name}: {e}")
                results[name] = None

        return results

    def add_metric(self, name: str, metric: Metric):
        """Add a metric to the collection."""
        self.metrics[name] = metric

    def remove_metric(self, name: str):
        """Remove a metric from the collection."""
        if name in self.metrics:
            del self.metrics[name]
