# tracealign/utils.py — Extended Utility Layer for I/O, Logging, Token Normalization, Token Stats
 
import os
import json
import logging
import re
from typing import List, Dict, Optional
 
logger = logging.getLogger("tracealign.utils")
logger.setLevel(logging.DEBUG)
 
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
 
 
def soft_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())
 
 
def load_token_probs(path: str) -> Dict[str, float]:
	logger.info(f"Loading token probability distribution from {path} ...")
	with open(path) as f:
    	probs = json.load(f)
	logger.info(f"Loaded {len(probs)} token probabilities.")
	return probs
 
 
def load_jsonl(path: str) -> List[Dict]:
	logger.info(f"Loading JSONL file: {path}")
	data = []
	with open(path, "r") as f:
    	for line in f:
        	data.append(json.loads(line))
	logger.info(f"Loaded {len(data)} records from {path}.")
	return data
 
 
def save_jsonl(path: str, records: List[Dict]):
	logger.info(f"Saving {len(records)} records to {path} ...")
	with open(path, "w") as f:
    	for r in records:
     	   f.write(json.dumps(r) + "\n")
	logger.info("Save complete.")
 
 
def read_corpus(dir_path: str) -> List[Dict[str, str]]:
	"""
	Loads all .txt/.jsonl files from a directory as documents
	"""
	corpus = []
	for filename in os.listdir(dir_path):
    	if filename.endswith(".txt"):
        	with open(os.path.join(dir_path, filename), "r") as f:
            	corpus.append({"id": filename, "text": f.read()})
    	elif filename.endswith(".jsonl"):
        	for record in load_jsonl(os.path.join(dir_path, filename)):
            	corpus.append(record)
	logger.info(f"Loaded corpus from {dir_path} with {len(corpus)} documents.")
	return corpus
 
 
def tokenize_corpus(corpus: List[Dict[str, str]], tokenizer_fn=soft_tokenize) -> List[Dict]:
	tokenized = []
	for doc in corpus:
    	tokens = tokenizer_fn(doc["text"])
    	tokenized.append({"id": doc["id"], "tokens": tokens})
	logger.info(f"Tokenized {len(tokenized)} documents.")
	return tokenized
 
 
def compute_token_frequencies(corpus: List[Dict[str, str]]) -> Dict[str, int]:
	freq = {}
	for doc in corpus:
    	tokens = soft_tokenize(doc["text"])
    	for tok in tokens:
        	freq[tok] = freq.get(tok, 0) + 1
	return freq
 
 
def normalize_token_frequencies(freq: Dict[str, int]) -> Dict[str, float]:
	total = sum(freq.values())
	return {k: v / total for k, v in freq.items()}
 
 
def save_token_distribution(path: str, freq: Dict[str, float]):
	with open(path, "w") as f:
    	json.dump(freq, f, indent=2)
	logger.info(f"Saved token distribution to {path}.")
