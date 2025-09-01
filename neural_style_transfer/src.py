import torch 
import torch.nn as nn
from torchvision import models 
from PIL import Image 
import torchvision.transforms as transforms

def load_image(image_path, transform=None,):
    image = Image.open(image_path).convert('RGB')

    if transform is not None:
        image = transform(image)
    
    image = image.unsqueeze(0)

    return image

class VGG(nn.Module):
    def __init__(self,vgg_model):
        super(VGG,self).__init__()
        self.features = [0,5,10,19,28]
        self.vgg = vgg_model
    
    def forward(self,x):
        features = []
        for layer_num,layer in enumerate(self.vgg):
            x = layer(x)

            if(layer_num in self.features):
                features.append(x)

        return features



def neural_style_transfer(content,style,content_weight,style_weight,num_steps,model,device,generated_image,optimizer):
    for step in range(num_steps):
        content_loss = style_loss = 0
        generated_features = model(generated_image)
        for content_layer,style_layer,gen_layer in zip(content,style,generated_features):
            b,c,h,w = gen_layer.size()
            gen_gram_matrix = gen_layer.reshape(c,h*w)
            generated_gram = gen_gram_matrix.mm(gen_gram_matrix.T)

            b,c,h,w = style_layer.size()
            style_gram_matrix = style_layer.reshape(c,h*w)
            style_gram = style_gram_matrix.mm(style_gram_matrix.T)

            content_loss += torch.mean((content_layer - gen_layer)**2)
            style_loss += torch.mean((generated_gram - style_gram)**2)

        total_loss = content_weight*content_loss + style_weight * style_loss 

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (step % 200 == 0):
            print(f"Step {step}, Total loss {total_loss.item()}")
            save_image(generated_image,'generated.png')

def show_image(generated_image):
    final_img = generated_image.cpu().detach().squeeze(0)
    final_img = transforms.ToPILImage()(final_img)
    final_img.show()

if __name__ == "__main__":
    vgg_model = models.vgg19(pretrained=True) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG(vgg_model.features)
    model.to(device).eval()
    trans = transforms.Compose([
        transforms.Resize((356,356)),
        transforms.ToTensor()
    ])

    org_image = load_image("annahathaway.png",transform=trans).detach().to(device)
    style_image = load_image("style.jpg",transform=trans).detach().to(device)
    generated_image = org_image.clone().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam([generated_image],lr = 0.001)
    content = [layer.detach() for layer in model(org_image)]
    style   =[layer.detach() for layer in model(style_image)]
    generated_image = neural_style_transfer(content,style,1,0.01,6000,model,device,generated_image,optimizer)
            
