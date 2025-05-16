# tests/test_traceindex.py — Unit Tests for SuffixArrayIndex
 
import unittest
from tracealign.traceindex import SuffixArrayIndex
 
class TestSuffixArrayIndex(unittest.TestCase):
	def setUp(self):
    	self.index = SuffixArrayIndex()
    	self.index.add_document("doc1", ["this", "is", "a", "test"])
    	self.index.add_document("doc2", ["this", "is", "another", "example"])
    	self.index.build()
 
	def test_match_basic(self):
    	result = self.index.match_span(["this", "is"])
    	self.assertTrue(any(r["doc_id"] == "doc1" for r in result))
 
	def test_match_no_result(self):
    	result = self.index.match_span(["nonexistent"])
    	self.assertEqual(len(result), 0)
 
if __name__ == '__main__':
	unittest.main()
