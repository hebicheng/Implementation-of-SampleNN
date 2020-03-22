import torch
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using cuda.')


D, d_in, d_h, d_out = 64,1000,100,10

# prepare dataset
x = torch.randn(D, d_in,device=device)
y = torch.randn(D, d_out,device=device)
# randomly initialize parameter
w1 = torch.randn(d_in, d_h,requires_grad=True,device=device)
w2 = torch.randn(d_h, d_out,requires_grad=True,device=device)

# 关于tensor.to的使用这里有一个坑
# 如果写为 w2 = torch.randn(d_h, d_out,requires_grad=True).to(device)
# 这样是无法计算w2的梯度的
# 此时的w2是一个复制到GPU的张量，所以w2已经不是叶子节点

# learning rate
learning_rate = 1e-6

# iteriter
for epoch in range(500):


    # forward pass
    out = x.mm(w1).clamp(min=0).mm(w2)
    # loss
    loss = (out - y).pow(2).sum()
    print(epoch, loss.item())

    # torch auto compute gradient
    loss.backward()

    # update parameter
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
