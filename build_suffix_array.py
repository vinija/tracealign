# scripts/build_suffix_array.py — Construct SuffixArrayIndex from Corpus
 
import os
import argparse
from tracealign.utils import read_corpus, tokenize_corpus
from tracealign.traceindex import SuffixArrayIndex
 
def main():
	parser = argparse.ArgumentParser(description="Build suffix array index for TRACEALIGN")
	parser.add_argument("--input_dir", required=True, help="Directory containing input .txt or .jsonl files")
	parser.add_argument("--output_path", required=True, help="Path to save the suffix array index")
	args = parser.parse_args()
 
	corpus = read_corpus(args.input_dir)
	tokenized = tokenize_corpus(corpus)
 
	index = SuffixArrayIndex()
	for doc in tokenized:
    	index.add_document(doc["id"], doc["tokens"])
	index.build()
	index.save(args.output_path)
 
if __name__ == "__main__":
	main()
 
