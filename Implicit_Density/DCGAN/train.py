import random
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
     

sample_dir=args.img_path
grid_size=64 # 8 x 8
fixed_noise = torch.randn(grid_size, input_latent, 1, 1, device=device)
num_epochs=5

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

for epoch in range(num_epochs):
    
    for i, data in enumerate(dataloader, 0):

        
        #<==========================================================>#

        Dis.zero_grad()
        
        real_cpu = data[0].to(device)

        b_size = real_cpu.size(0)

       
        label = torch.full((b_size,), 1, device=device)
        
        output = Dis(real_cpu).view(-1)
        
        errD_real = criterion(output, label)
        
        errD_real.backward()

        D_x = output.mean().item()

        
        noise = torch.randn(b_size, input_latent, 1, 1, device=device)

        fake = Gen(noise)
        label.fill_(0)

        output = Dis(fake.detach()).view(-1)

        errD_fake = criterion(output, label)

        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        
        
        Dis_optimizer.step()

        #<==========================================================>#

        Gen.zero_grad()
        label.fill_(1)  

        output = Dis(fake).view(-1)

        errG = criterion(output, label)

        errG.backward()
        D_G_z2 = output.mean().item()

        Gen_optimizer.step()

        if i % 200 == 0:
            print('[%d/%d][%d/%d]      Loss_D: %.4f      Loss_G: %.4f      D(x): %.4f      D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = Gen(fixed_noise).detach().cpu()
            img_list.append( torchvision.utils.make_grid(fake, padding=2, normalize=True))



           
            torchvision.utils.save_image(fake, os.path.join(sample_dir, 'fake_images-{}.png'.format(iters+1)))
        iters += 1
