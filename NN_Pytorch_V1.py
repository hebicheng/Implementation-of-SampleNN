import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using cuda.')
D, d_in, d_h, d_out = 64,1000,100,10

# prepare dataset
x = torch.randn(D, d_in).to(device)
y = torch.randn(D, d_out).to(device)

# randomly initialize parameter
w1 = torch.randn(d_in, d_h).to(device)
w2 = torch.randn(d_h, d_out).to(device)

# learning rate
learning_rate = 1e-6

# iteriter
for epoch in range(500):

    # forward pass
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    out = h_relu.mm(w2)

    # loss
    loss = (out - y).pow(2).sum().item()
    print(epoch, loss)

    # backprop to compute gradients of w1 and w2 with respect to loss
    grad_out = 2.0 * (out - y)
    grad_w2 = h_relu.T.mm(grad_out)
    grad_h_relu = grad_out.mm(w2.T)
    grad_h_relu[h<0] = 0
    grad_w1 = x.T.mm(grad_h_relu)

    # update parameter
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

