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
            


class DataProcessor():

    def __init__(self,vocab,sequences , targets ):
        
        self.vocab = vocab
        self.sequences , self.targets = sequences,targets 
        self.N = len(sequences)

    def convert(self,sequence):
        return LongTensor(self.vocab(sequence))
    
    def sample(self,batch_size):
        X,Y = [],[]
        for i in range(batch_size):
            index = random.randint(0,self.N-1)
            seq,targ = self.sequences[index] , self.targets[index]
            X.append(self.convert(seq))
            Y.append(targ)
        X = pad_sequence(X,padding_value=0,batch_first=True,)
        Y = FloatTensor(Y)
        return X,Y

if __name__ == '__main__':
    from vocab import Vocab 
    vocab = Vocab(['a','b','c'])
    dp = DataProcessor(vocab,['aa','bb','cc','aaa','bbb','ccc'],[1,2,3,1,2,3])
    x,y =  dp.sample(3)
    print(x)
    print(y)