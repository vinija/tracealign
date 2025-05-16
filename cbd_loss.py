# tracealign/cbd_loss.py — CBD-Aware DPOTrainer Extension for Alignment-Preserving Fine-Tuning
 
import torch
import logging
from trl import DPOTrainer
from transformers import PreTrainedTokenizer
from typing import List, Dict
 
logger = logging.getLogger("tracealign.cbd")
logger.setLevel(logging.INFO)
 
class CBDDPOTrainer(DPOTrainer):
	def __init__(self, model, ref_model, args, tokenizer: PreTrainedTokenizer,
             	tracer, bci_model, bci_threshold: float, lambda_penalty: float = 0.01):
    	super().__init__(model=model, ref_model=ref_model, args=args, tokenizer=tokenizer)
    	self.tracer = tracer
    	self.bci = bci_model
    	self.threshold = bci_threshold
    	self.lambda_penalty = lambda_penalty
 
	def compute_cbd_penalty(self, decoded: List[str]) -> float:
    	risky = 0.0
    	matches = self.tracer.trace_completion(decoded)
    	for m in matches:
        	if "span" in m and self.bci.high_risk(m["span"], self.threshold):
            	risky += max(0, self.bci.compute_bci(m["span"]) - self.threshold)
    	return risky
 
	def compute_loss(self, model, inputs, return_outputs=False):
    	loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
 
    	# Decode model output for CBD penalty
    	logits = outputs.logits  # [batch, seq_len, vocab_size]
    	batch_penalty = 0.0
    	for b in range(logits.size(0)):
        	predicted_ids = logits[b].argmax(dim=-1).tolist()
        	tokens = self.tokenizer.convert_ids_to_tokens(predicted_ids, skip_special_tokens=True)
        	cbd_penalty = self.compute_cbd_penalty(tokens)
        	batch_penalty += cbd_penalty
        	logger.debug(f"CBD penalty for sample {b}: {cbd_penalty:.4f}")
 
    	total_loss = loss + self.lambda_penalty * batch_penalty
    	logger.info(f"CBD loss added: {self.lambda_penalty * batch_penalty:.4f}, Total loss: {total_loss:.4f}")
 
    	return (total_loss, outputs) if return_outputs else total_loss
