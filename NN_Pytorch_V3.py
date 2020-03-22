import torch
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using cuda.')


D, d_in, d_h, d_out = 64,1000,100,10

# prepare dataset
x = torch.randn(D, d_in,device=device)
y = torch.randn(D, d_out,device=device)

# 定义一个神经网络
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=d_in, out_features= d_h),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=d_h,out_features= d_out),
)
model.to(device)

loss_fn = torch.nn.MSELoss(reduction='sum')
# learning rate
learning_rate = 1e-4

# iteriter
for epoch in range(500):


    # forward pass
    out = model(x)
    # loss
    loss = loss_fn(out,y)
    print(epoch, loss.item())
    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # torch auto compute gradient
    loss.backward()

    # update parameter
    with torch.no_grad():
        for parameter in model.parameters():
            parameter -= learning_rate * parameter.grad