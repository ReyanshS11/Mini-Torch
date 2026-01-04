import numpy as np
from tensor import Tensor

def backward(tns: Tensor) -> None:
        if tns.data.ndim != 0:
            raise RuntimeError(
                "grad can be implicitly created only for scalar outputs"
            )
        
        topo = []
        visited = set()

        def build(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(tns)

        if tns.grad is None:
            tns.grad = np.ones_like(tns.data)

        for t in reversed(topo):
            t._backward()