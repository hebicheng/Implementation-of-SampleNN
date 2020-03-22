import numpy as np

D, d_in, d_h, d_out = 64,1000,100,10

# prepare dataset
x = np.random.randn(D, d_in)
y = np.random.randn(D, d_out)

# randomly initialize parameter
w1 = np.random.randn(d_in, d_h)
w2 = np.random.randn(d_h, d_out)

# learning rate
learning_rate = 1e-6

# iteriter
for epoch in range(500):

    # forward pass
    h = x.dot(w1)
    h_relu = np.maximum(0,h)
    out = h_relu.dot(w2)

    # loss
    loss = np.square(out - y).sum()
    print(epoch, loss)

    # backprop to compute gradients of w1 and w2 with respect to loss
    grad_out = 2.0 * (out - y)
    grad_w2 = h_relu.T.dot(grad_out)
    grad_h_relu = grad_out.dot(w2.T)
    grad_h_relu[h<0] = 0
    grad_w1 = x.T.dot(grad_h_relu)

    # update parameter
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

