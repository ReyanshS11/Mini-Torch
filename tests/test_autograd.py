import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_torch.core.tensor import Tensor

class TestAutogradBasic(unittest.TestCase):

    def test_add_scalar(self):
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)

        z = x + y
        z.backward()

        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, 1.0)

    def test_sub_scalar(self):
        x = Tensor(5.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)

        z = x - y
        z.backward()

        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, -1.0)

    def test_mul_scalar(self):
        x = Tensor(4.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)

        z = x * y
        z.backward()

        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, 4.0)

    def test_div_scalar(self):
        x = Tensor(6.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)

        z = x / y
        z.backward()

        self.assertEqual(x.grad, 0.5)
        self.assertEqual(y.grad, -1.5)


class TestSum(unittest.TestCase):

    def test_sum_backward(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        y = x.sum()
        y.backward()

        np.testing.assert_array_equal(
            x.grad,
            np.array([1.0, 1.0, 1.0])
        )

    def test_square_sum(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        y = x * x
        z = y.sum()
        z.backward()

        np.testing.assert_array_equal(
            x.grad,
            np.array([2.0, 4.0, 6.0])
        )


class TestGraphStructure(unittest.TestCase):

    def test_shared_subgraph(self):
        x = Tensor(2.0, requires_grad=True)

        y = x * x
        z = y + y
        z.backward()

        # z = 2 * x^2 → dz/dx = 4x
        self.assertEqual(x.grad, 8.0)

    def test_chain_rule(self):
        x = Tensor(3.0, requires_grad=True)

        y = x * x      # x^2
        z = y * x      # x^3
        z.backward()

        # dz/dx = 3x^2 = 27
        self.assertEqual(x.grad, 27.0)


class TestBackwardErrors(unittest.TestCase):

    def test_backward_non_scalar_raises(self):
        x = Tensor([1.0, 2.0], requires_grad=True)

        y = x * x

        with self.assertRaises(RuntimeError):
            y.backward()


if __name__ == "__main__":
    unittest.main()