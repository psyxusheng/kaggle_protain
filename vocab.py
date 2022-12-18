import os

path = os.path.dirname(os.path.abspath(__file__))

class Vocab():
    
    def __init__(self,vocab_list = []):
        self.load(vocab_list)

    def from_file(self, vocab_file = None):

        if vocab_file is None:
            vocab_file = os.path.join(path,'vocab.dat')
        vocab_list = open(vocab_file,'r').read().split()
        self.load(vocab_list)


    def load(self,vocab_list):

        self._word_to_id = {word:i+1 for i,word in enumerate(vocab_list)}
        self._id_to_word = {i+1:word for i,word in enumerate(vocab_list)}
        self._word_to_id['[pad]'] = 0
        self._id_to_word[0] = '[pad]'


    
    def convert(self,sequence):
        return [self._word_to_id.get(s,0) for s in sequence]
    
    def convert_back(self,sequence):
        return [self._id_to_word.get(s,'unk') for s in sequence]

    def __call__(self, sequence):
        return self.convert(sequence)

if __name__ == '__main__':
    vocab_list = ['a','b','c','d','e']
    vocab = Vocab(vocab_list)
    print(vocab._id_to_word)