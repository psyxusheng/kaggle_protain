from torch import from_numpy
import numpy as np
import pandas as pd

def get_data(filename):
    data = pd.read_csv(filename)
    return data['protein_sequence'].values , data['tm'].values 

def load_ptvectors(filename,add_pad = True):
    vectors = np.load(filename)
    vs = vectors.shape[1]
    if add_pad :
        zeros = np.zeros([1,vs])
        vectors = np.concatenate([zeros , vectors],axis=0)
    return from_numpy(vectors).float()

if __name__ == '__main__':
    import numpy as np
    output = load_ptvectors('./vectors.npy')
    print(output.shape)