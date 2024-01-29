import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

print('PyTorch version:', torch.__version__)
print('torchvision verseion:', torchvision.__version__)
print('Is GPU avaibale:', torch.cuda.is_available())

# settings
batchsize = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tf = transforms.Compose([transforms.ToTensor(), 
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



mnist_train = datasets.MNIST(root = '../../data/MNIST',
                                 train = True,
                                 transform = tf,
                                 download = False)
mnist_validation = datasets.MNIST(root = '../../data/MNIST',
                                      train = False,
                                      transform = tf)

mnist_train_loader = DataLoader(mnist_train, batch_size = batchsize, shuffle = True)
mnist_validation_loader = DataLoader(mnist_validation, batch_size = batchsize, shuffle = False)

print('the number of training data', len(mnist_train))
print('the number of validation data', len(mnist_validation))





#############################################################
############## 1x1 invertible convolution　##################
#############################################################
class Invertible1x1Conv(nn.Module):
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
      
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        
        if torch.det(W) < 0:
            W[:,0] = -W[:,0]
        
        self.conv.weight.data = W.view(c, c, 1)
        
    def forward(self, z, reverse=False):
        batch_size, group_size, n_of_group = z.size()
        
        W = self.conv.weight.squeeze(2)
        
        if reverse:
            if not hasattr(self, 'w_inverse'):
                W_inverse = W.inverse() 
                W_inverse = torch.autograd.Variable(W_inverse.unsqueeze(2)) 
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            log_det_W = batch_size * n_of_group * torch.logdet(W)
            return self.conv(z), log_det_W





class NN(nn.Module):
    def __init__(self, n_in_channels, n_h_channels, n_out_channels):
        super(NN, self).__init__()
        self.cv1 = nn.Conv1d(n_in_channels, n_h_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(n_h_channels)
        self.cv2 = nn.Conv1d(n_h_channels, n_h_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(n_h_channels)
        self.cv3 = nn.Conv1d(n_h_channels, n_out_channels, kernel_size=3, stride=1, padding=1)
        self.cv3.weight.data.zero_()
        self.cv3.bias.data.zero_()
        
    def forward(self, forward_input):
        out = F.relu(self.bn1(self.cv1(forward_input)))
        out = F.relu(self.bn2(self.cv2(out)))
        return self.cv3(out)






class Glow(nn.Module):
    def __init__(self, n_flows, n_group, n_early_every, n_early_size, affine=True):
        super(Glow, self).__init__()
        
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_of_group = None
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.n_h_channels = 32
        self.affine = affine
        
        self.NN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        
        n_half = int(n_group/2)
        
        n_remaining_channels = n_group
        
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
                
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            
            # affine coupling
            if self.affine:
                self.NN.append(NN(n_half, self.n_h_channels, 2*n_half))
            # additive coupling
            else:
                self.NN.append(NN(n_half, self.n_h_channels, n_half))
                
        self.n_remaining_channels = n_remaining_channels
        
    def forward(self, forward_input):
        assert(forward_input.size(1) % self.n_group == 0)
        self.n_of_group = int(forward_input.size(1) / self.n_group) # グループごとの点数
        
        image = forward_input.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_image = []
        log_s_list = []
        log_det_W_list = []
        
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_image.append(image[:,:self.n_early_size,:])
                image = image[:,self.n_early_size:,:]
                
            image, log_det_W = self.convinv[k](image)
            log_det_W_list.append(log_det_W)
            
            n_half = int(image.size(1)/2)
            image_0 = image[:,:n_half,:]
            image_1 = image[:,n_half:,:]
            
            output = self.NN[k](image_0)
            if self.affine:
                log_s = output[:,:n_half,:]
                b = output[:,n_half:,:]
                image_1 = torch.exp(log_s)*image_1 + b
                log_s_list.append(log_s)
            else:
                # b = output
                image_1 = image_1 + output
                log_s_list.append(0)
                
            image = torch.cat([image_0, image_1], dim=1)
            
        output_image.append(image)
        return torch.cat(output_image,dim=1), log_s_list, log_det_W_list
        

    def infer(self, n_sample, sigma=1.0):
        assert(self.n_of_group is not None)
        image = torch.cuda.FloatTensor(n_sample, self.n_remaining_channels, self.n_of_group).normal_()
        image = torch.autograd.Variable(sigma*image) 
        
        for k in reversed(range(self.n_flows)):
            n_half = int(image.size(1)/2)
            image_0 = image[:,:n_half,:]
            image_1 = image[:,n_half:,:]
            
            output = self.NN[k](image_0)
            if self.affine:
                log_s = output[:,:n_half,:]
                b = output[:,:n_half,:]
                image_1 = (image_1 - b) / torch.exp(log_s)
            else:
                # b = output
                image_1 = image_1 - output
            
            image = torch.cat([image_0, image_1], dim=1)
            
            image = self.convinv[k](image, reverse=True)
            
            if k % self.n_early_every == 0 and k > 0:
                z = torch.cuda.FloatTensor(n_sample, self.n_early_size, self.n_of_group).normal_()
                image = torch.cat([sigma*z, image], dim=1)
        
        image = image.permute(0,2,1).contiguous().view(n_sample, -1).data
        return image


#############################################################
####################### GLOW Loss　##########################
#############################################################      
def GlowLoss(glow_output, sigma=1.0):
    z, log_s_list, log_det_W_list = glow_output
    for i, log_s in enumerate(log_s_list):
        if i == 0:
            log_s_total = torch.sum(log_s)
            log_det_W_total = log_det_W_list[i]
        else:
            log_s_total += torch.sum(log_s)
            log_det_W_total += log_det_W_list[i]
          
    loss = torch.sum(z*z)/(2*sigma*sigma) - log_s_total - log_det_W_total
    return loss/(z.size(0)*z.size(1)*z.size(2))
net = Glow(6, 8, 2, 2)
net = net.to(device)

learning_rate = 0.0001
optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)
# optimizer = optim.Adam(net.parameters(), lr = learning_rate)

num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('The number of parameters:', num_trainable_params)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-1-8c465fc5860b> in <module>()
----> 1 net = Glow(6, 8, 2, 2)
      2 net = net.to(device)
      3 
      4 learning_rate = 0.0001
      5 optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)

NameError: name 'Glow' is not defined
def train(train_loader):
    net.train()
    running_loss = 0
    
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        
        glow_output = net(inputs)
        loss = GlowLoss(glow_output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    train_loss = running_loss / len(train_loader)
    
    return train_loss
def validation(validation_loader, epoch):
    net.eval()
    running_loss = 0
    n_sample = 25
    sigma = 1.0
    sample_every = 1
    output_dir = '../../data/glow_MNIST'
    
    with torch.no_grad():
        for inputs, _ in validation_loader:
            inputs = inputs.to(device)
            inputs = inputs.view(inputs.size(0), -1)
  
            glow_output = net(inputs)
            loss = GlowLoss(glow_output)
            
            running_loss += loss.item()
            
    validation_loss = running_loss / len(validation_loader)
    
    if epoch % sample_every == 0:
        sampled_images = net.infer(n_sample, sigma)
        sampled_images = sampled_images.view(n_sample, 1, 28, 28)
        save_image(sampled_images.data.cpu(), '{}/{}.png'.format(output_dir, epoch), nrow=5, padding=1)
    
    return validation_loss
train_loss_list = []
validation_loss_list = []

n_epochs = 100
for epoch in range(n_epochs):
    train_loss = train(mnist_train_loader)
    validation_loss = validation(mnist_validation_loader, epoch)
    
    train_loss_list.append(train_loss)
    validation_loss_list.append(validation_loss)
    
    print('epoch[%d/%d] train_loss:%1.4f validation_loss:%1.4f' % (epoch+1, n_epochs, train_loss, validation_loss) )
epoch[1/100] train_loss:-0.81


