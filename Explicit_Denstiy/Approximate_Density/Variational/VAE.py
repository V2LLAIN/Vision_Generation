import torch
import torch.nn as nn
import torch.Functional as F

class VAE(nn.Module):
    def __init__(self):
        super (VAE, self).__init_()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear (400, 20)
        self.fc22 = nn.Linear (400, 20)
        self.fc3 = nn.Linear (20, 400)
        self. fc4 = nn.Linear (400, 784)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function (recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy (recon_x, x.view(-1, 784), size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD​
