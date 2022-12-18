import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np
import os

class ConvProtein(nn.Module):

    def __init__(self,embeddings, hidden_sizes = [32,64,128,256,512] , lr = 1e-3):
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

    def forward(self, inputs):
        # inputs is in shape [batch_size , 60 , 60 , 1]
        # transpose : [batch , width , length , dim] --> [batch , dim , width , length]
        out = self._embedding(inputs).transpose(3,1)
        for conv in self.conv_layers:
            out = conv(out)
        out = F.adaptive_max_pool2d(out,(1,1))
        out = out.squeeze(-1).squeeze(-1)
        out = self._mlp(out)
        out = out.squeeze(-1)
        return out 
    
    def train(self,x,y):

        preds = self(x)
        loss = F.mse_loss(preds,y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._steps += 1

        if self._steps % 100 == 0:
            print(f'{self._steps:6d}--{loss.to("cpu").numpy().item():.3f}')

    def save(self,name = None):
        if name is None :
            sizes = '_'.join(map(str,self.hidden))
            name = f'ckpt.{self.n_vocab}-{self.emb_dim}-{sizes}.pkl'
        if not os.path.exists('./ckpt'):
            os.path.makedirs('./ckpt')
        torch.save(self.state_dict(),os.path.join('./ckpt',name))
        print(f'saved parameters to {name}')
    
    def load(self,name):
        self.load_state_dict(torch.load(os.path.join('./ckpt',name)))
        print(f'loaded model from {name}')
