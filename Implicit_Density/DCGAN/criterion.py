criterion = nn.BCELoss()

Gen_optimizer = torch.optim.Adam(Gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
Dis_optimizer = torch.optim.Adam(Dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
     
