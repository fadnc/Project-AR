from __future__  import annotations
import math

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
    
    def backward(self):
        topo = []
        visited = set()
        
        def build(v):
            if v not in visited:
                visited.add(v)
                
                for child in v._prev:
                    build(child)
                    
                topo.append(v)
                
        build(self)
        
        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()
        
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        out  = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), "-")
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        out = Value(-self.data, (self,), "neg")
        
        def _backward():
            self.grad += -1 * out.grad
            
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), "/")
        
        def _backward():
            self.grad += 1/other.data * out.grad
            other.grad += -self.data/other.data**2 * out.grad
        
        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** power, (self, ), "**")
        
        def _backward():
            self.grad += power * (self.data ** (power-1) * out.grad)
                                              
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLu")
        
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
            
        out._backward = _backward
        return out
    
a = Value(2)
b = Value(3)

c = a * b
d = c + a

d.backward()

print(a.grad)
print(b.grad)