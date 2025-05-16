# scripts/decode_with_prov.py — Decode using ProvDecode Penalty
 
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tracealign.prov_decode import ProvDecode
from tracealign.traceindex import SuffixArrayIndex
from tracealign.bci import BeliefConflictIndex
 
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True)
	parser.add_argument("--suffix_index", required=True)
	parser.add_argument("--token_probs", required=True)
	parser.add_argument("--prompt", required=True)
	parser.add_argument("--threshold", type=float, default=10.0)
	parser.add_argument("--gamma", type=float, default=1.0)
	args = parser.parse_args()
 
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	model = AutoModelForCausalLM.from_pretrained(args.model)
	index = SuffixArrayIndex()
	index.load(args.suffix_index)
	token_probs = BeliefConflictIndex.load_token_probs(args.token_probs)
	bci = BeliefConflictIndex(token_probs)
	prov = ProvDecode(index, bci, bci_threshold=args.threshold, gamma=args.gamma)
 
	inputs = tokenizer(args.prompt, return_tensors="pt")
	logits = model(**inputs).logits[0, -1]
	vocab = [tokenizer.decode([i]) for i in range(len(logits))]
	adjusted = prov.adjust_logits(args.prompt.split(), vocab, logits.tolist())
	top_k = sorted(zip(vocab, adjusted), key=lambda x: -x[1])[:10]
	for token, score in top_k:
    	print(f"{token}\t{score:.3f}")
 
if __name__ == "__main__":
	main()
