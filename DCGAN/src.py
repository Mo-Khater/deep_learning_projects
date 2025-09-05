import torch
import torch.nn as nn
import torch.optim as optimizer
import torchvision.utils as vutils
import torchvision.datasets as datasets
from torchvision import transforms as transforms
from torch.utils.data import DataLoader

class Discriminator(nn.Module):
    def __init__(self,in_channels,features_d = 64):
        """
            in_channels : the channels of the input image
            feature_d : is a constant value that is used to get output channels
        """
        super(Discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(in_channels,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(.2),
            self.block(in_channels=features_d,out_channels=features_d*2,kernel_size=4,stride=2,padding=1),
            self.block(in_channels=features_d*2,out_channels=features_d*4,kernel_size=4,stride=2,padding=1),
            self.block(in_channels=features_d*4,out_channels=features_d*8,kernel_size=4,stride=2,padding=1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.dis(x)

    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2)
        )

class Generator(nn.Module):
    def __init__(self,z_dim,in_channels,features_g = 64):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            self.block(z_dim,features_g *16, kernel_size=4,stride=1,padding=0),
            self.block(features_g *16,features_g *8, kernel_size=4,stride=2,padding=1),
            self.block(features_g *8,features_g *4, kernel_size=4,stride=2,padding=1),
            self.block(features_g *4,features_g *2, kernel_size=4,stride=2,padding=1),
            nn.ConvTranspose2d(features_g*2,in_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )

    def block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.gen(x)


def load_celebA_dataset(path="./data", batch_size=128, image_size=64):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)])  # Normalize RGB
    ])
    dataset = datasets.CelebA(root=path, split="train", download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

def load_MNIST_dataset(batch_size = 32):
    transform = transforms.Compose(
    [
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,.5,.5), (0.5,.5,.5)),
    ]
    )

    dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def init_weights(model):
    for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hyperparameters
num_epochs = 10
z_dim = 100
img_dim = 64
channel_img = 3
lr = 2e-4
batch_size = 32

loader = load_celebA_dataset()
gen = Generator(z_dim,channel_img).to(device)
dis = Discriminator(channel_img).to(device)
gen.apply(init_weights)
dis.apply(init_weights)
criterion = nn.BCELoss()
optmG = torch.optim.Adam(gen.parameters(),lr = lr,betas=(0.5, 0.999))
optmD = torch.optim.Adam(dis.parameters(),lr = lr,betas=(0.5, 0.999))

for epoch in range(num_epochs):

    for idx , (real_imgs,_) in enumerate(loader):
        noise = torch.randn((real_imgs.size(0),z_dim)).to(device)
        fake_imgs = gen(noise.view(-1,z_dim,1,1))

        # discriminator loss maximize : 1/m * [sum(log(D(xi)))] + 1/n * [sum(log(1 - D(G(zi))))]
        # generator loss maximize : 1/n * [sum(log(D(G(zi))))]
        # real_imgs = real_imgs.view(-1,img_dim).to(device)
        real_imgs = real_imgs.to(device)
        dis_real_outputs = dis(real_imgs)
        dis_fake_ouputs = dis(fake_imgs.detach())
        dis_real_loss = criterion(dis_real_outputs.view(real_imgs.size(0),-1),torch.ones((real_imgs.size(0),1)).to(device))
        dis_fake_loss = criterion(dis_fake_ouputs.view(real_imgs.size(0),-1),torch.zeros((real_imgs.size(0),1)).to(device))
        dis_total_loss = dis_fake_loss + dis_real_loss

        optmD.zero_grad()
        dis_total_loss.backward()
        optmD.step()

        dis_fake_ouputs = dis(fake_imgs)
        gen_loss = criterion(dis_fake_ouputs.view(real_imgs.size(0),-1),torch.ones((real_imgs.size(0),1)).to(device))
        optmG.zero_grad()
        gen_loss.backward()
        optmG.step()

        if idx % 200 == 0:
            with torch.no_grad():
                test_noise = torch.randn(16, z_dim).to(device)
                fake_samples = gen(test_noise.view(16,z_dim,1,1))
                vutils.save_image(fake_samples, f"epoch{epoch}_batch{idx}.png",
                                normalize=True, nrow=4)