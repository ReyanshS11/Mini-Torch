import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_torch.core.tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_add_basic(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x + y
        np.testing.assert_array_equal(z.data, np.array([4.0, 6.0]))

    def test_add_broadcast(self):
        x = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        y = Tensor([10.0, 20.0, 30.0], requires_grad=True)
        z = x + y
        loss = z.sum()
        loss.backward()
        # Check gradients broadcast correctly
        np.testing.assert_array_equal(x.grad, np.ones_like(x.data) * 3)
        np.testing.assert_array_equal(y.grad, np.ones_like(y.data) * 3)

    def test_sub_mul_div(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = Tensor([4.0, 6.0], requires_grad=True)
        z = ((y - x) * x) / y
        loss = z.sum()
        loss.backward()
        # Manually calculate gradients
        grad_x = np.array([0.0, 0.0])
        grad_y = np.array([0.25, 0.25])
        np.testing.assert_array_almost_equal(x.grad, grad_x)
        np.testing.assert_array_almost_equal(y.grad, grad_y)

    def test_mean(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.mean()
        y.backward()
        np.testing.assert_array_equal(x.grad, np.full((2,2), 0.25))

    def test_reshape(self):
        x = Tensor([[1,2,3],[4,5,6]], requires_grad=True)
        y = x.reshape((3,2))
        loss = y.sum()
        loss.backward()
        np.testing.assert_array_equal(x.grad, np.ones_like(x.data))

    def test_scalar_operations(self):
        x = Tensor(2.0, requires_grad=True)
        y = x + 3
        z = y * 2
        z.backward()
        self.assertEqual(x.grad, 2.0)

    def test_chain_operations(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = ((x + y)**2 - x) / y
        loss = z.sum()
        loss.backward()
        # Check gradients are numeric
        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(y.grad.shape, y.data.shape)

    def test_high_dim_tensor(self):
        x = Tensor(np.ones((2,3,4)), requires_grad=True)
        y = x.mean()
        y.backward()
        np.testing.assert_array_equal(x.grad, np.ones_like(x.data) / x.data.size)

    def test_empty_tensor(self):
        x = Tensor(np.array([]), requires_grad=True)
        y = x.sum()
        y.backward()
        np.testing.assert_array_equal(x.grad, np.array([]))

    def test_broadcast_mul(self):
        x = Tensor([[1.0],[2.0],[3.0]], requires_grad=True)
        y = Tensor([10.0,20.0], requires_grad=True)
        z = x * y
        loss = z.sum()
        loss.backward()
        np.testing.assert_array_equal(x.grad, np.array([[30],[30],[30]]))
        np.testing.assert_array_equal(y.grad, np.array([6,6]))

if __name__ == "__main__":
    unittest.main()