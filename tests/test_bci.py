# tests/test_bci.py — Unit Tests for BeliefConflictIndex
 
import unittest
from tracealign.bci import BeliefConflictIndex
 
class TestBeliefConflictIndex(unittest.TestCase):
	def setUp(self):
    	self.bci = BeliefConflictIndex({"a": 0.1, "b": 0.01, "c": 0.001})
 
	def test_compute_bci(self):
    	score = self.bci.compute_bci(["a", "b"])
    	self.assertGreater(score, 0)
 
	def test_normalized(self):
    	density = self.bci.normalized_bci(["a", "b"])
    	self.assertLess(density, 100)
 
	def test_high_risk(self):
    	risky = self.bci.high_risk(["b", "c"], threshold=10.0)
    	self.assertTrue(risky)
 
if __name__ == '__main__':
	unittest.main()
