# scripts/eval_traceshield.py — Evaluate Traceshield Refusal on a Prompt Dataset
 
import argparse
from tracealign.utils import load_jsonl
from tracealign.traceindex import SuffixArrayIndex
from tracealign.bci import BeliefConflictIndex
from tracealign.traceshield import TraceShield
 
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--prompts", required=True, help="JSONL file of prompts")
	parser.add_argument("--index", required=True, help="Path to suffix array index")
	parser.add_argument("--probs", required=True, help="Token probability JSON")
	parser.add_argument("--threshold", type=float, default=10.0)
	args = parser.parse_args()
 
	prompts = load_jsonl(args.prompts)
	token_probs = BeliefConflictIndex.load_token_probs(args.probs)
 
	index = SuffixArrayIndex()
	index.load(args.index)
	bci = BeliefConflictIndex(token_probs)
	shield = TraceShield(index, bci, args.threshold)
 
	for p in prompts:
    	text = p["completion"]
    	tokens = text.strip().split()
    	result = shield.explain(tokens)
    	print("===", p["id"])
    	print(result)
 
if __name__ == "__main__":
	main()
 
