from __future__  import annotations

class Value:
    """
    a scalar value that supports automatic differentiation
    """

    def __init__(self, data: float, _children=(), _op=""):
        
        self.data = data
        
        #gradient accumulated during backprop
        self.grad = 0.0
        
        #operation that produced this node
        self._op = _op
        
        #parent nodes
        self._prev = set(_children) # prev is written to understand the parent nodes of the current node
        
        #fn executed during backward phase
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value(data={self.data}, grad = {self.grad})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out
    
    def __mul__(self, other):
        out  = Value(self.data * other.data, (self, other), "*")
        return out
    
a = Value(2.0)
b = Value(3.0)

c = a * b
d = c + a
e = d * b

print(e)