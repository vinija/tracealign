# tests/test_traceshield.py — Unit Tests for TraceShield Refusal Logic
 
import unittest
from tracealign.traceshield import TraceShield
from tracealign.traceindex import SuffixArrayIndex
from tracealign.bci import BeliefConflictIndex
 
class DummyTracer:
	def trace_span(self, span):
    	return [{"span": span}]
 
class TestTraceShield(unittest.TestCase):
	def setUp(self):
    	dummy_probs = {"bad": 0.00001, "safe": 0.9}
    	self.bci = BeliefConflictIndex(dummy_probs)
    	self.tracer = DummyTracer()
    	self.shield = TraceShield(self.tracer, self.bci, threshold=10.0)
 
	def test_refuse_true(self):
    	tokens = ["this", "is", "bad"]
    	self.assertTrue(self.shield.refuse(tokens))
 
	def test_refuse_false(self):
    	tokens = ["safe", "tokens"]
    	self.assertFalse(self.shield.refuse(tokens))
 
if __name__ == '__main__':
	unittest.main()
