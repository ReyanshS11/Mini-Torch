    y = Tensor([[10,20,30,40]], requires_grad=True)

    z = x + y
    loss = z.sum()
    loss.backward()

    print(x.grad)  # [[4], [4], [4]]
    print(y.grad)  # [[3, 3, 3, 3]]