# tracealign/bci.py — Belief Conflict Index (BCI) Computation with Rarity, Density, and Entropy Views
 
import math
import numpy as np
import logging
from typing import List, Dict, Optional
 
logger = logging.getLogger("tracealign.bci")
logger.setLevel(logging.DEBUG)
 
class BeliefConflictIndex:
	def __init__(self, token_probs: Dict[str, float], default_prob: float = 1e-9):
    	self.token_probs = token_probs
    	self.default_prob = default_prob
    	self.entropy_cache: Dict[str, float] = {}
    	self._build_entropy_cache()
 
	def _build_entropy_cache(self):
    	logger.info("Building entropy cache for BCI...")
    	for token, p in self.token_probs.items():
        	self.entropy_cache[token] = -math.log(p + 1e-12)
    	logger.info(f"Cached entropy for {len(self.entropy_cache)} tokens.")
 
    def compute_bci(self, span: List[str]) -> float:
    	"""Raw BCI score: negative log-likelihood over pretraining distribution"""
    	return sum(self.entropy_cache.get(t, -math.log(self.default_prob)) for t in span)
 
	def normalized_bci(self, span: List[str]) -> float:
    	"""BCI density: per-token risk density"""
    	return self.compute_bci(span) / max(1, len(span))
 
	def compute_kl_divergence(self, span: List[str]) -> float:
    	"""KL divergence of span against unigram prior P_train"""
    	span_probs = {}
    	for t in span:
        	span_probs[t] = span_probs.get(t, 0) + 1.0 / len(span)
    	kl = 0.0
    	for t, pt in span_probs.items():
        	q = self.token_probs.get(t, self.default_prob)
        	kl += pt * math.log(pt / q)
    	return kl
 
	def compute_entropy(self, span: List[str]) -> float:
    	return -sum(math.log(self.token_probs.get(t, self.default_prob)) for t in span) / max(1, len(span))
 
	def max_token_risk(self, span: List[str]) -> float:
    	return max(self.entropy_cache.get(t, -math.log(self.default_prob)) for t in span)
 
	def high_risk(self, span: List[str], threshold: float) -> bool:
    	return self.compute_bci(span) > threshold
 
	def explain_span(self, span: List[str]) -> Dict:
    	"""Return detailed diagnostics for a span"""
    	entropy_scores = [self.entropy_cache.get(t, -math.log(self.default_prob)) for t in span]
    	return {
        	"span": span,
        	"total_bci": round(self.compute_bci(span), 4),
        	"density": round(self.normalized_bci(span), 4),
        	"max_token_risk": round(self.max_token_risk(span), 4),
        	"kl_divergence": round(self.compute_kl_divergence(span), 4),
        	"per_token_entropy": entropy_scores
    	}
 
	def compare_spans(self, a: List[str], b: List[str]) -> Dict[str, float]:
    	"""Compare two spans for relative BCI statistics"""
    	return {
        	"bci_a": self.compute_bci(a),
        	"bci_b": self.compute_bci(b),
 	       "delta_bci": self.compute_bci(b) - self.compute_bci(a),
        	"kl_a": self.compute_kl_divergence(a),
        	"kl_b": self.compute_kl_divergence(b)
    	}
 
	def rank_spans(self, spans: List[List[str]], mode: str = "bci") -> List[Tuple[int, float]]:
    	score_fn = {
        	"bci": self.compute_bci,
        	"density": self.normalized_bci,
        	"entropy": self.compute_entropy,
        	"kl": self.compute_kl_divergence
    	}[mode]
    	return sorted(((i, score_fn(span)) for i, span in enumerate(spans)), key=lambda x: -x[1])
