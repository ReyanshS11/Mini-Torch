import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad

        self.grad = None
        self._prev = set()
        self._backward = lambda: None

    def numerical_grad(self, f, x, eps=1e-6):
        return (f(x + eps) - f(x - eps)) / (2 * eps)

    def __add__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad - out.grad if other.grad is not None else -out.grad # type: ignore
        
        out._backward = _backward
        return out
    
    def __mul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad * other.data if self.grad is not None else out.grad * other.data
            if other.requires_grad:
                other.grad = other.grad + out.grad * self.data if other.grad is not None else out.grad * self.data

        out._backward = _backward
        return out
    
    def __truediv__(self, other: Tensor) -> Tensor:
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                grad_x = out.grad / other.data
                self.grad = self.grad + grad_x if self.grad is not None else grad_x
            if other.requires_grad:
                grad_y = -out.grad * self.data / (other.data ** 2) # type: ignore
                other.grad = other.grad + grad_y if other.grad is not None else grad_y
        
        out._backward = _backward
        return out
    
    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad - out.grad if self.grad is not None else -out.grad # type: ignore

        out._backward = _backward
        return out
    
    def __pow__(self, power) -> Tensor:
        assert isinstance(power, (int, float)), "only scalar powers supported"
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                grad = power * (self.data ** (power - 1)) * out.grad # type: ignore
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out
    
    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                grad = out.grad * np.ones_like(self.data)
                self.grad = self.grad + grad if self.grad is not None else grad
            
        out._backward = _backward
        return out
    
    def backward(self):
        if self.data.ndim != 0:
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

        build(self)

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        for t in reversed(topo):
            t._backward()

    def shape(self):
        return self.data.shape
    
if __name__ == "__main__":
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * x
    z = y.sum()
    z.backward()

    print(x.grad)  # [2, 4, 6]