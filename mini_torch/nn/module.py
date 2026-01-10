import numpy as np

class Module:
    def parameters(self):
        params = []
        for v in vars(self).values():
            if hasattr(v, "requires_grad") and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
        
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)