import os

path = os.path.dirname(os.path.abspath(__file__))

class Vocab():
    def __init__(self,vocab_file = None):
        if vocab_file is None:
            vocab_file = os.path.join(path,'vocab.dat')
        vocab = open(vocab_file,'r').read().split()
        self._word_to_id = {word:i+1 for i,word in enumerate(vocab)}
        self._id_to_word = {i+1:word for i,word in enumerate(vocab)}
    def convert(self,sequence):
        return [self._word_to_id.get(s,0) for s in sequence]
    def convert_back(self,sequence):
        return [self._id_to_word.get(s,'unk') for s in sequence]

if __name__ == '__main__':
    vocab = Vocab()
    print(vocab._id_to_word)