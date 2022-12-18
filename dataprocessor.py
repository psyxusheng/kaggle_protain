import random 
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor , FloatTensor

def padding(sequences,padding_idx = 0 , max_len = None):
    lengths = [len(seq) for seq in sequences]
    if max_len is None:
        max_len = max(lengths)
    ret = []
    for l,seq in zip(lengths,sequences):
        if l > max_len :
            start = random.randint(0,l-max_len)
            sub = seq[start : start + max_len]
            ret.append([s for s in sub])
        else:
            ret.append([s for s in seq] + [padding_idx]*(max_len-l))
    return ret 

def trunc_or_extend(seq,max_len):
    N = len(seq)
    if N > max_len:
        start = random.randint(0,N-max_len)
        sub = seq[start : start + max_len]
        return sub 
    else:
        # copy and expand 
        return seq[:] + [0]*(max_len - N)
            


class DataProcessor():

    def __init__(self,vocab, sequences , targets , targ_range = None):
        
        self.vocab = vocab
        self.sequences , self.raw_targets = sequences,targets 
        
        if targ_range is None:
            targ_range = [min(targets),max(targets)]
        self.targ_range = targ_range
        self.min_v , self.max_v = targ_range


        self.targets = [self.downscale(v) for v in targets]
        self.N = len(sequences)
    
    def upscale(self,v):
        return (v+1)/2 * (self.max_v - self.min_v) + self.min_v
            
    def downscale(self ,  v):
        return ((v - self.min_v) / (self.max_v - self.min_v)) * 2 - 1 

    def convert(self,sequence):
        return LongTensor(self.vocab(sequence))
    
    def sample(self,batch_size,max_len = 3600,shape=[60,60]):
        X,Y = [],[]
        for i in range(batch_size):
            index = random.randint(0,self.N-1)
            seq,targ = self.sequences[index] , self.targets[index]
            token_ids = self.vocab(seq)
            X.append(trunc_or_extend(token_ids,max_len))
            Y.append(targ)
        X = LongTensor(X).reshape([batch_size,*shape])
        Y = FloatTensor(Y)
        return X,Y

if __name__ == '__main__':
    from vocab import Vocab 
    vocab = Vocab(['a','b','c'])
    dp = DataProcessor(vocab,['aa','bb','cc','aaa','bbb','ccc'],[1,2,3,1,2,3])
    print(dp.targ_range)
    x,y =  dp.sample(3)
    print(x.shape)
    print(y)