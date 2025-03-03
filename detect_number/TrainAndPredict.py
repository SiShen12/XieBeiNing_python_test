#导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

#数据预处理
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
#加载训练集和测试集
train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
#创建数据加载器
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)
#定义简单的CNN模型




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        x = F.relu(nn.functional.max_pool2d(self.conv1(x),2))
        x = F.relu(nn.functional.max_pool2d(self.conv2(x),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
#初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#训练模型
def train(model,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target)in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch:{}[{}/{}({:.0f})%]\tLoss:{:.6f}'.format(epoch,
                batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/
                len(train_loader),loss.item()))

#测试模型
def test(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            output = model(data)
            test_loss += criterion(output,target).item()
            pred =output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)
    print(f'\nTest set:Average loss:{test_loss:.4f},Accuracy:{correct}'
          f'/{len(test_loader.dataset)}({100.*correct/len(test_loader.dataset):.0f}%)\n')

#训练和测试模型
for epoch in range(1,5):
    train(model,train_loader, optimizer, epoch)
    test(model,test_loader)