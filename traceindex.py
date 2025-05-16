# tracealign/traceindex.py — Efficient and Attributable Suffix Array Index for Unsafe Span Retrieval
 
from typing import Tuple
import pickle
 
class SuffixArrayIndex:
	def __init__(self):
    	self.suffix_array: List[Tuple[List[str], str, int]] = []  # (suffix tokens, doc_id, start index)
    	self.lexicon: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    	self.doc_lengths: Dict[str, int] = {}
 
	def add_document(self, doc_id: str, tokens: List[str]):
    	self.doc_lengths[doc_id] = len(tokens)
    	for i in range(len(tokens)):
        	suffix = tokens[i:]
        	self.suffix_array.append((suffix, doc_id, i))
        	self.lexicon[tokens[i]].append((doc_id, i))
 
	def build(self):
    	logger.info("Sorting suffix array lexicographically...")
    	self.suffix_array.sort(key=lambda x: x[0])
    	logger.info(f"Suffix array built with {len(self.suffix_array)} suffixes.")
 
	def match_span(self, span: List[str], top_k: int = 5) -> List[Dict]:
    	matches = []
    	low, high = 0, len(self.suffix_array)
    	while low < high:
        	mid = (low + high) // 2
        	if self.suffix_array[mid][0][:len(span)] < span:
        	    low = mid + 1
        	else:
            	high = mid
    	start = low
    	while start < len(self.suffix_array):
        	suffix, doc_id, offset = self.suffix_array[start]
        	if suffix[:len(span)] == span:
            	matches.append({"doc_id": doc_id, "position": offset, "span": suffix[:len(span)]})
            	if len(matches) >= top_k:
                	break
            	start += 1
        	else:
            	break
    	return matches
 
	def save(self, path: str):
    	logger.info(f"Saving suffix array index to {path}")
    	with open(path, "wb") as f:
        	pickle.dump({
            	"suffix_array": self.suffix_array,
            	"lexicon": self.lexicon,
            	"doc_lengths": self.doc_lengths
        	}, f)
    	logger.info("Suffix array saved.")
 
	def load(self, path: str):
    	logger.info(f"Loading suffix array index from {path}")
    	with open(path, "rb") as f:
        	data = pickle.load(f)
        	self.suffix_array = data["suffix_array"]
        	self.lexicon = data["lexicon"]
        	self.doc_lengths = data["doc_lengths"]
    	logger.info("Suffix array loaded successfully.")
