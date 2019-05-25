#This is a practice from https://morvanzhou.github.io/tutorials/machine-learning/torch/3-01-regression/
#And mainly describe about How to built an easy example for machine learning
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # built intent
y = x.pow(2) + 0.2*torch.rand(x.size())                 #noise point

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
    
net=Net(n_feature=1,n_hidden=15,n_output=1)

print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.2)
loss_func=torch.nn.MSELoss()

plt.ion()
for t in range(200000):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

        
plt.ioff()
plt.show()
