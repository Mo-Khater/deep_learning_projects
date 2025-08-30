from get_dataloader import get_loader
import torch
from torchvision import transforms
from models import CNNtoRNN
from utils import load_checkpoint,save_checkpoint
# load data 

transform = transforms.Compose(
    [transforms.Resize((299, 299)), transforms.ToTensor(),]
)
loader, dataset = get_loader(
    "/content/drive/MyDrive/colab/images/", "/content/drive/MyDrive/colab/captions.txt", transform=transform
)

# hyperparameters
embed_size = 256
hidden_size = 256
learning_rate = .0001
vocabulary = dataset.vocab
vocab_size = len(vocabulary)
num_layers = 1
num_epochs = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def train_model(load_model = False,save_model = False):
    model = CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers)
    model.to(device) # Move model to device BEFORE initializing optimizer
    criterian = torch.nn.CrossEntropyLoss(ignore_index=vocabulary.stoi["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    step = 0
    if load_model :
        step = load_checkpoint(torch.load('/content/my_checkpoint.pth.tar'),model,optimizer)

    model.train()
    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                'state_dict':model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step' : step
            }
            save_checkpoint(checkpoint)

        for idx,(imgs,captions) in enumerate(loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs,captions[:-1])
            loss = criterian(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if idx % 50 == 0:
              print(f"Epoch [{epoch+1}/{num_epochs}] Step [{idx}/{len(loader)}] Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train_model(save_model=True,load_model=True)