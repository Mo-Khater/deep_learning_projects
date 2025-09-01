import torch 
from torch import nn as nn 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils

class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super(Discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.01),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    
    def forward(self,x): 
        return self.dis(x)
    

class Generator(nn.Module):
    def __init__(self,z_dim,img_dim):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(.01),
            nn.Linear(256,img_dim),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)
    
def load_MNIST_dataset(batch_size = 32):
    transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    )

    dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hyperparameters
num_epochs = 50
z_dim = 64
img_dim = 28*28*1 
lr = 1e-4
batch_size = 32

loader = load_MNIST_dataset(batch_size)
gen = Generator(z_dim,img_dim).to(device)
dis = Discriminator(img_dim).to(device)
lossG = nn.BCELoss()
lossD = nn.BCELoss()
optmG = torch.optim.Adam(gen.parameters(),lr = lr)
optmD = torch.optim.Adam(dis.parameters(),lr = lr)

for epoch in range(num_epochs):
    
    for idx , (real_imgs,_) in enumerate(loader):
        noise = torch.randn((batch_size,z_dim)).to(device)
        fake_imgs = gen(noise)

        # discriminator loss maximize : 1/m * [sum(log(D(xi)))] + 1/n * [sum(log(1 - D(G(zi))))]
        # generator loss maximize : 1/n * [sum(log(D(G(zi))))]
        real_imgs = real_imgs.view(-1,img_dim).to(device)
        dis_real_outputs = dis(real_imgs)
        dis_fake_ouputs = dis(fake_imgs.detach())
        dis_real_loss = lossG(dis_real_outputs,torch.ones((batch_size,1)).to(device))
        dis_fake_loss = lossG(dis_fake_ouputs,torch.zeros((batch_size,1)).to(device))
        dis_total_loss = dis_fake_loss + dis_real_loss 

        optmD.zero_grad()
        dis_total_loss.backward()
        optmD.step()

        dis_fake_ouputs = dis(fake_imgs)
        gen_loss = lossG(dis_fake_ouputs,torch.ones((batch_size,1)).to(device))
        optmG.zero_grad()
        gen_loss.backward()
        optmG.step()

    if idx % 200 == 0:
        with torch.no_grad():
            test_noise = torch.randn(16, z_dim).to(device)
            fake_samples = gen(test_noise).view(-1, 1, 28, 28)
            vutils.save_image(fake_samples, f"epoch{epoch}_batch{idx}.png", 
                            normalize=True, nrow=4)
        


