# tracealign/prov_decode.py — Provenance-Aware Logit Filtering for Safer Decoding
 
import logging
from typing import List
 
logger = logging.getLogger("tracealign.prov")
logger.setLevel(logging.INFO)
 
class ProvDecode:
	def __init__(self, tracer, bci_model, bci_threshold: float, gamma: float = 1.0, context_window: int = 8):
  	  self.tracer = tracer
    	self.bci = bci_model
    	self.threshold = bci_threshold
    	self.gamma = gamma
    	self.window = context_window
 
	def extract_span(self, prefix: List[str], candidate: str) -> List[str]:
    	"""Combine prefix window and candidate token to form a traceable span"""
    	context = prefix[-self.window:] if len(prefix) >= self.window else prefix
    	return context + [candidate]
 
	def compute_risk(self, span: List[str]) -> float:
    	matches = self.tracer.trace_span(span)
    	if not matches:
        	return 0.0
    	risk_scores = [self.bci.compute_bci(m['span']) for m in matches if 'span' in m]
    	return max(risk_scores) if risk_scores else 0.0
 
	def veto(self, prefix: List[str], token: str) -> bool:
    	span = self.extract_span(prefix, token)
    	risk = self.compute_risk(span)
    	if risk > self.threshold:
        	logger.debug(f"Token '{token}' vetoed due to BCI={risk:.2f} > {self.threshold:.2f}")
        	return True
    	return False
 
	def adjust_logits(self, prefix: List[str], vocab: List[str], logits: List[float]) -> List[float]:
    	"""
    	Penalize logits for high-risk tokens during decoding.
       
    	Parameters:
        	prefix: Current generation context
        	vocab: List of candidate tokens
        	logits: List of original token logits
 
    	Returns:
        	Modified logits penalizing unsafe tokens
    	"""
    	adjusted = []
    	for i, tok in enumerate(vocab):
        	span = self.extract_span(prefix, tok)
        	risk = self.compute_risk(span)
        	penalty = self.gamma if risk > self.threshold else 0.0
        	adjusted.append(logits[i] - penalty)
    	return adjusted
 
	def rank_tokens(self, prefix: List[str], vocab: List[str]) -> List[tuple]:
    	"""Sort tokens by their provenance risk score"""
    	scores = [(tok, self.compute_risk(self.extract_span(prefix, tok))) for tok in vocab]
    	return sorted(scores, key=lambda x: -x[1])
