# scripts/run_cbd_training.py — Finetune a Model with CBD Loss
 
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOConfig
from tracealign.cbd_loss import CBDDPOTrainer
from tracealign.traceindex import SuffixArrayIndex
from tracealign.bci import BeliefConflictIndex
 
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True)
	parser.add_argument("--ref_model", type=str, required=True)
	parser.add_argument("--token_probs", type=str, required=True)
	parser.add_argument("--suffix_index", type=str, required=True)
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--threshold", type=float, default=10.0)
	args = parser.parse_args()
 
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	model = AutoModelForCausalLM.from_pretrained(args.model)
	ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model)
 
	token_probs = BeliefConflictIndex.load_token_probs(args.token_probs)
	bci = BeliefConflictIndex(token_probs)
	index = SuffixArrayIndex()
	index.load(args.suffix_index)
 
	trainer = CBDDPOTrainer(
    	model=model,
    	ref_model=ref_model,
    	tokenizer=tokenizer,
    	args=TrainingArguments(
        	output_dir=args.output,
        	per_device_train_batch_size=2,
        	evaluation_strategy="no",
        	save_strategy="epoch",
        	num_train_epochs=3,
        	learning_rate=5e-5
    	),
    	tracer=index,
    	bci_model=bci,
    	bci_threshold=args.threshold,
    	lambda_penalty=0.01
	)
 
	trainer.train()
	trainer.save_model(args.output)
 
if __name__ == "__main__":
	main()
