import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# load MNIST dataset
train_dataset = datasets.MNIST('/files/', train=True, download=True,
                            transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]))

test_dataset = datasets.MNIST('/files/', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]))

# initialize all variables
fraction = 0.3
iteration = 5
num_clients = 5
batch_size = 32
stepsize = 0.01

# CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class Client:
    def __init__(self, model, train_loader, criterion, stepsize):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.stepsize = stepsize
        self.size = len(self.train_loader.dataset)
    
    def calc_gradient(self, model):
        model.train()
        
        # initialize gradient dictionary
        grad={}
        for name, param in self.model.named_parameters():
            grad[name]=torch.zeros_like(param)
            
        
        for images, labels in self.train_loader:
            model.zero_grad()
            outputs = model(images)
            loss= self.criterion(outputs, labels)
            loss.backward()
            
            # accumulate gradients
            for name, param in model.named_parameters():
                grad[name]+=param.grad*len(labels)
            
           
        # normalize by averaging gradient
        for name in grad:
            grad[name]/=self.size
            
        return grad
        
    def updates(self, global_model, full_gradient):      
        # give global model weights to all clients
        self.model.load_state_dict(global_model.state_dict())
        
        # initialize local model
        local_model = {}
        for name, param in self.model.named_parameters():
            local_model[name] = param.clone()
        
        # initialize local step-size
        samples = sum(len(c.train_loader.dataset) for c in clients)
        local_ss = self.stepsize*self.size/samples
        
        # iterate over the min-batches
        for images, labels in self.train_loader:
            self.model.zero_grad()
            outputs = self.model(images)
            loss= self.criterion(outputs, labels)
            loss.backward()
            
            # SVRG update
            for name, param in self.model.named_parameters():
                grad_local_model = param.grad
                grad_global_model = self.calc_gradient(global_model)[name]
                theta = grad_local_model - grad_global_model + full_gradient[name]
                local_model[name] -= local_ss*theta
                
                
        # get trained model's weight
        local_weight=self.model.state_dict(local_model)
        
        return local_weight, self.size


# compute full gradient on global model
def calc_full_gradient(global_model, clients):
    # get total size
    total_size = sum(client.size for client in clients)
    
    # initialize dictionary for full gradient
    full_gradient={}
    for name, param in global_model.named_parameters():
        full_gradient[name] = torch.zeros_like(param)
    
    for client in clients:
        # calculate the gradient of clients
        local_gradient = client.calc_gradient(global_model)
        # accumulate gradients
        for name in full_gradient:
            full_gradient[name]+=local_gradient[name]*client.size
    
    # average the gradients from all clients
    for name in full_gradient:
        full_gradient[name] /= total_size
    
    return full_gradient


def fed_svrg(global_model, clients):
    for i in range(iteration):
        # calc full gradient on global model
        full_gradient = calc_full_gradient(global_model, clients)
        
        # select clients
        m = max(int(fraction*len(clients)),1)
        selected_clients = random.sample(clients,m)
        
        # client updates
        weights={}
        for name, param in global_model.named_parameters():
            weights[name] = torch.zeros_like(param)
        
        sizes = 0
                
        for client in selected_clients:
            local_weight, local_size = client.updates(global_model, full_gradient)
            for name in weights:
                weights[name]+=local_weight[name]*local_size
            
            sizes+=local_size
            
        for name in weights:
            weights[name]/=sizes
        
        global_model.load_state_dict(weights)
    
    return global_model
    
# testing
def test(net, test_loader):
     criterion = nn.CrossEntropyLoss()
     net.eval()  # set model to evaluation mode
     running_loss = 0.0
     test_correct = 0
     test_total = 0
     for images, labels in test_loader:
         outputs = net(images) # get raw outputs (logits)
         loss = criterion(outputs, labels) # calculate loss

         running_loss += loss.item() * images.size(0)
         _, predicted = torch.max(outputs.data, 1)
         test_total += labels.size(0)
         test_correct += (predicted == labels).sum().item()

     # print test loss and accuracy
     test_loss = running_loss / len(test_loader.dataset)
     test_accuracy = test_correct / test_total
     #print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
     return test_loss, test_accuracy
           
    
def initialize_clients(num_clients, train_datasets, global_model, batch_size, stepsize):
    clients=[]
    for i in range(num_clients):
        local_model = Net()
        local_model.load_state_dict(global_model.state_dict())
        train_loader=DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        client=Client(local_model, train_loader, criterion, stepsize)
        clients.append(client)
        
    return clients


global_model = Net()

c_train_datasets = random_split(train_dataset, [len(train_dataset)//num_clients]*num_clients)
clients = initialize_clients(num_clients, c_train_datasets, global_model, batch_size, stepsize)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
global_model = fed_svrg(global_model, clients)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss, test_accuracy = test(global_model, test_loader)
print(f"Global Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")