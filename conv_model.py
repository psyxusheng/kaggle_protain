import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np
import os

class ConvProtein(nn.Module):

    def __init__(self,embeddings, hidden_sizes = [32,64,128,256,512] , lr = 1e-3 , use_cuda = False):
        # say input shape of this protein is in shape [batch_size,100,100,1] 
        # last dim is the acid thing 
        # first embedding it
        nn.Module.__init__(self)

        if isinstance(embeddings , (list,tuple)):   
            n_vocab , emb_dim,*_ = embeddings
            weights = None
        else:
            if type(embeddings).__name__ == 'ndarray':
                embeddings = torch.from_numpy(embeddings).float()

            n_vocab , emb_dim = embeddings.size()
            weights = embeddings
        
        self._embedding = nn.Embedding(n_vocab , emb_dim ,padding_idx=0, _weight = weights)

        self.n_vocab , self.emb_dim = n_vocab , emb_dim
        self.hidden = hidden_sizes
        
        conv_layers = []
        inp_feat    = emb_dim
        for hs in hidden_sizes:
            layer = nn.Sequential(nn.Conv2d(inp_feat , hs*2 , kernel_size=3,stride=1,padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(hs*2, hs , kernel_size=3,stride=2,padding=1))
            inp_feat = hs
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        self._mlp = nn.Sequential(nn.Linear(hidden_sizes[-1] , 100 ), nn.ReLU() , nn.Linear(100,1),nn.Tanh())


        self.optimizer = torch.optim.Adam(self.parameters(),lr = lr)

        self._steps = 0

        self.device = 'cuda' if use_cuda and torch.cuda.is_available()  else "cpu"


    def forward(self, inputs,training=True):
        # inputs is in shape [batch_size,  60 , 60 ]
        if self.device == 'cuda':
            inputs = inputs.to(self.device)
        out = self._embedding(inputs).transpose(-1,1)
        if training:
            out = F.dropout(out,training=training)
        for conv in self.conv_layers:
            out = conv(out)
        out = F.adaptive_max_pool2d(out,(1,1))
        if training:
            out = F.dropout(out,training=training)
        out = out.squeeze(-1).squeeze(-1)
        out = self._mlp(out)
        out = out.squeeze(-1)
        return out 
    
    def updates(self,x,y):

        preds = self(x,training = True)
        if self.device =='cuda':
            y = y.to(self.device) 

        loss = F.mse_loss(preds,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._steps += 1

        if self._steps % 100 == 0:
            print(f'{self._steps:6d}--{loss.to("cpu").data.numpy().item():.3f}')
        
        if self._steps % 1000 == 0:
            self.save()

    def predicts(self,batch):
        with torch.no_grad():
            out = self(batch,training = False)
        out = out.to('cpu').numpy().tolist()
        return out

    def save(self,name = None):
        if name is None :
            sizes = '_'.join(map(str,self.hidden))
            name = f'ckpt.{self.n_vocab}-{self.emb_dim}-{sizes}-{self._steps}.pkl'
        if not os.path.exists('./ckpt'):
            os.makedirs('./ckpt')
        torch.save(self.state_dict(),os.path.join('./ckpt',name))
        print(f'saved parameters to {name}')
    
    def load(self,name):
        self.load_state_dict(torch.load(os.path.join('./ckpt',name),map_location=torch.device(self.device)))
        print(f'loaded model from {name}')

if __name__ == '__main__':
    a = '123.123123123123.456'
    b = a.split('.')
    print(b)