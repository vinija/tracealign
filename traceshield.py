# tracealign/traceshield.py — Inference-Time Refusal Filter with Span Tracing, Risk Reporting, and Attribution Logging
 
import logging
from typing import List, Dict
 
logger = logging.getLogger("tracealign.traceshield")
logger.setLevel(logging.DEBUG)
 
class TraceShield:
	def __init__(self, tracer, bci_model, threshold: float, window_size: int = 8, max_matches: int = 5):
    	self.tracer = tracer
    	self.bci = bci_model
        self.threshold = threshold
    	self.window_size = window_size
    	self.max_matches = max_matches
 
	def _window_spans(self, tokens: List[str]) -> List[List[str]]:
    	return [tokens[i:i + self.window_size] for i in range(len(tokens) - self.window_size + 1)]
 
	def detect_risky_spans(self, tokens: List[str]) -> List[Dict]:
    	risky = []
    	for span in self._window_spans(tokens):
        	matches = self.tracer.trace_span(span)
        	for match in matches[:self.max_matches]:
            	if self.bci.high_risk(match['span'], self.threshold):
                	report = self.bci.explain_span(match['span'])
                	report.update({
                    	"match_doc": match.get("doc_id", "?"),
      	              "match_offset": match.get("position", -1)
                	})
                	risky.append(report)
    	return risky
 
	def refuse(self, tokens: List[str]) -> bool:
    	for span in self._window_spans(tokens):
        	matches = self.tracer.trace_span(span)
        	for match in matches[:self.max_matches]:
            	if self.bci.high_risk(match['span'], self.threshold):
                	logger.warning(f"TRACESHIELD: Refusing output due to high-BCI span: {match['span']}")
                	return True
    	return False
 
	def explain(self, tokens: List[str]) -> Dict:
    	risky = self.detect_risky_spans(tokens)
    	return {
        	"refused": len(risky) > 0,
        	"span_count": len(tokens) - self.window_size + 1,
        	"risky_count": len(risky),
        	"risky_spans": risky[:5]  # only top few for verbosity
    	}
 
	def detailed_log(self, tokens: List[str]):
    	logger.info("Running TRACESHIELD diagnostic log...")
    	risky = self.detect_risky_spans(tokens)
    	for r in risky:
        	logger.info(f"BCI {r['total_bci']} | Span: {' '.join(r['span'])} | KL: {r['kl_divergence']} | Max Token Risk: {r['max_token_risk']} | Matched in: {r['match_doc']} @ {r['match_offset']}")
 
	def refusal_report(self, tokens: List[str]) -> str:
    	risky = self.detect_risky_spans(tokens)
    	if not risky:
        	return "✅ Output passed TRACESHIELD. No high-BCI spans detected."
    	report = ["⛔ REFUSAL TRIGGERED BY TRACESHIELD"]
    	for r in risky:
        	report.append(f"- Span: {' '.join(r['span'])} | BCI: {r['total_bci']} | Source: {r['match_doc']} @ {r['match_offset']}")
    	return "\n".join(report)
