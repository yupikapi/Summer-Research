import torch
import torch.nn as nn
import torch.optim as optim
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
epochs = 5
num_clients = 5
batch_size = 32

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
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Client:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
    
    def train(self, global_model):
        
        # give global model weights to all clients
        self.model.load_state_dict(global_model.state_dict())
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.11)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        local_size = 0
        
        for e in range(epochs):
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels) #compute loss
                loss.backward() # calc gradients
                optimizer.step() # adjust weight
                
                # update data size
                local_size+=len(labels)
        
        # get trained model's weight
        local_weight = self.model.state_dict()
        
        return local_weight, local_size


def fed_avg(global_model, clients):
    # initialize global_model parameters (weights) to zero
    global_dict = global_model.state_dict()
    
    for i in range(iteration):
        # create empty dictionary to store weights
        weights = {}
        for key, val in global_dict.items():
            weights[key] = torch.zeros_like(val)
        
        # initialize data sizes to accumulate client data size
        sizes = 0
        
        # select clients
        m = max(int(fraction*len(clients)),1)
        selected_clients = random.sample(clients,m)

        for client in selected_clients:
            # retrieve model weight and client's data size (update client)
            local_weight, local_size = client.train(global_model)       
            # accumulate weights of model params
            for key in weights.keys():
                weights[key] += local_weight[key]*local_size        
            # accumulate size of client's dataset
            sizes += local_size        
        # update global weights
        for key in global_dict.keys():
            global_dict[key] = weights[key]/sizes
        
        # apply the updated weiths to global_model
        global_model.load_state_dict(global_dict)       
            
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
    

def initialize_clients(num_clients, train_datasets, global_model, batch_size):
    clients=[]
    for i in range(num_clients):
        local_model = Net()
        local_model.load_state_dict(global_model.state_dict())
        train_loader=DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True)
        client=Client(local_model, train_loader)
        clients.append(client)
        
    return clients

global_model = Net()

c_train_datasets = random_split(train_dataset, [len(train_dataset)//num_clients]*num_clients)
clients = initialize_clients(num_clients, c_train_datasets, global_model, batch_size)

global_model = fed_avg(global_model, clients)


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss, test_accuracy = test(global_model, test_loader)
print(f"Global Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")