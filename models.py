import torch
from torch import nn
from torchvision import models

class encoderCNN(nn.Module):
    def __init__(self,embed_size , train_CNN = False):
        super(encoderCNN,self).__init__()
        self.inception = models.inception_v3(pretrained = True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)

        for name , param in self.inception.named_parameters():
            if name in ["fc.weight","fc.bias"]:
                param.requires_grad  = True
            elif not train_CNN:
                param.requires_grad  = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.5)

    def forward(self,images):
        features = self.inception(images)
        # Check if the output is an InceptionOutputs named tuple and access the logits
        if isinstance(features, models.inception.InceptionOutputs):
            features = features.logits
        features = self.dropout(self.relu(features))
        return features
        # features (batch_size,embed_size)

class decoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers,train_RNN = True , teaching_force_ratio = 1):
        super(decoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(.5)
        self.teaching_force_ratio = teaching_force_ratio
        self.train_RNN = train_RNN

    def forward(self,features,captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim = 0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
        # outputs (seq_length,batch_size,vocab_size)


class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers,train_RNN= False,teaching_force_ratio = 1,train_CNN =False):
        super(CNNtoRNN,self).__init__()
        self.encoderCNN = encoderCNN(embed_size,train_CNN)
        self.decoderRNN = decoderRNN(embed_size,hidden_size,vocab_size,num_layers,train_RNN,teaching_force_ratio)

    def forward(self,images,captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features,captions)
        return outputs
        # outputs (seq_length,batch_size,vocab_size)

    def caption_image(self,image,max_length = 50,vocabulary = None):
        self.eval() # Set model to evaluation mode
        input = self.encoderCNN(image.unsqueeze(0))
        # input (1,embed_size)

        decoder_LSTM = self.decoderRNN.lstm
        decoder_linear = self.decoderRNN.linear
        result = []
        states = None
        with torch.no_grad():
            for _ in range(max_length):
                lstm_output,states = decoder_LSTM(input.unsqueeze(0),states)
                output = decoder_linear(lstm_output)
                # output (1,1,voc_size)
                predicted = output.argmax(dim=2)
                token_idx = predicted.item()
                result.append(token_idx)
                input = self.decoderRNN.embed(predicted).squeeze(0)
                if (vocabulary.itos[token_idx] == '<EOS>'):
                    break 
        
        return [vocabulary.itos[idx] for idx in result]
