from torch.utils.data import Dataset
import torch
import numpy as np
class DataTransformDict():
    def __init__(self):
        self.tense2num = {'sp': 0, 'tp': 1, 'pg': 2, 'p': 3}
        self.alpha2num = {'SOS': 0, 'EOS': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11,
                     'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22,
                     'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27}
        self.num2alpha = dict((v,k) for k,v in self.alpha2num.items()) #exchange key and value
        self.num2tense = dict((v,k) for k,v in self.tense2num.items()) #exchange key and value

class TenseTransformDataset(Dataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.words_list = []
        self.tenses_list = []
        if mode == 'train':
            words_list = np.loadtxt(path, dtype=np.str)
            self.tenses_list = ['sp', 'tp', 'pg', 'p']*len(words_list)  # simple present(sp), third person(tp), present progressive(pg), simple past(p); tenses_list: ['sp', 'tp', 'pg', 'p', 'sp', 'tp', 'pg', 'p', ...]
            self.words_list = words_list.flatten().tolist() #words_list: ['abandon', 'abandons', 'abandoning', 'abandoned', 'abet', 'abets', 'abetting', 'abetted', ...
        elif mode == 'test':
            self.tenses_list = [['sp','p'], ['sp','pg'], ['sp','tp'], ['sp','tp'], ['p','tp'], ['sp','pg'], ['p','sp'], ['pg','sp'], ['pg','p'], ['pg','tp']]
            self.words_list = np.loadtxt(path, dtype=np.str).tolist()
            #for line in fin: #line: 'abandon abandoned\n'
                #words = line.strip('\n').split(' ') #words: ['abandon', 'abandoned']
                #self.words_list.append(words) #words_list: [['abandon', 'abandoned'], ['abet', 'abetting'], ...]
    def __len__(self): #return the size of dataset
        return len(self.words_list)
    def __getitem__(self, index): #return processed index-th words and tenses
        typeTrans = DataTransformDict()
        alpha2num = typeTrans.alpha2num
        tense2num = typeTrans.tense2num
        if self.mode == 'train':
            word = self.words_list[index] #word: 'abandon'
            tense = self.tenses_list[index] #tense: 'sp'
            vector = [alpha2num[char] for char in word] #vector :[2, 3, 2, 15, 5, 16, 15]
            vector.insert(0, alpha2num['SOS'])
            vector.append(alpha2num['EOS']) #vector: [0, 2, 3, 2, 15, 5, 16, 15, 1]
            #vector = vector + [1]*(20 - len(vector)) #padding
            pair = (torch.tensor(vector, dtype=torch.long), torch.tensor(tense2num[tense], dtype=torch.long)) #pair: (tensor([ 0,  2,  3,  2, 15,  5, 16, 15,  1]), tensor(0))
            return pair
        elif self.mode == 'test':
            pair_list = []
            for word, tense in zip(self.words_list[index], self.tenses_list[index]): #word: 'abandon', tense: 'sp', self.words_list[index]: ['abandon', 'abandoned'], self.tenses_list[index]: ['sp', 'p']
                # convert word to vector
                vector = [alpha2num[char] for char in word] #vector :[2, 3, 2, 15, 5, 16, 15]
                vector.insert(0, alpha2num['SOS'])
                vector.append(alpha2num['EOS']) #vector: [0, 2, 3, 2, 15, 5, 16, 15, 1]

                # append vector version's (word, tense) pair to pair_list
                pair = ( torch.tensor(vector,dtype=torch.long), torch.tensor(tense2num[tense],dtype=torch.long) ) #pair: (tensor([ 0,  2,  3,  2, 15,  5, 16, 15,  1]), tensor(0))
                pair_list.append(pair)
            return tuple(pair_list) #tuple(pair_list): ((tensor([ 0,  2,  3,  2, 15,  5, 16, 15,  1]), tensor(0)), (tensor([ 0,  2,  3,  2, 15,  5, 16, 15,  6,  5,  1]), tensor(3)))

#dataset = TenseTransformDataset("./Lab4_dataset/train.txt", "train")