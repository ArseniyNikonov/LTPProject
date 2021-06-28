__author__ = "Barbara Plank (original Keras version). Adapted to PyTorch by Antonio Toral"

"""
Exercise 3. A simple feedforward NN for animacy classification with an embedding layer
"""
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
print(torch.__version__) # this should be at least 1.0.0
#torch.set_printoptions(profile="full") # print complete tensors rather than truncated, for debugging


# parser = argparse.ArgumentParser()
# parser.add_argument('train', help="animacy data training file")
# parser.add_argument('dev', help="animacy data dev file")
# parser.add_argument('test', help="animacy data test file")
# parser.add_argument('--iters', help="epochs (iterations)", type=int, default=10)
# parser.add_argument('--embed_dim', help="dimension of the embedding layer (0 = no embedding layer)", type=int, default=0)
# args = parser.parse_args()

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ----------------------------------------------------------
# Classes/functions related to PyTorch

class NN(nn.Module):
    """ The neural network that will be used """

    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): size of the input features (i.e. vocabulary size)
            output_dim (int): number of classes
        """
    
    
        #For any comments explaining code from the previopus assignment, see the previous assignment
        super(NN, self).__init__()
        self.embedding = nn.Embedding(input_dim, 300)
        self.lstm=nn.LSTM(300, 8, num_layers=1, dropout=0, bidirectional=True )
        self.fc=nn.Linear(16, output_dim)
        

    def forward(self, x):
        """The forward pass of the NN

        Args:
            x (torch.Tensor): an input data tensor. 
                x.shape should be (batch_size, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch_size, num_classes)
        """
        sentence=x
        x = self.embedding(x)
        x=torch.mean(x,dim=1)
        x, _ = self.lstm(x.view(len(sentence),1,-1))
        x = self.fc(x.view(len(sentence),-1))

        x = F.log_softmax(x, dim=1)      
        return x
    
    def print_params(self):
        """Print the parameters (theta) of the network. Mainly for debugging purposes"""
        for name, param in model.named_parameters():
            print(name, param.data)
    
    def get_embedding(self, word_id):
        """Print the embedding vector for an input word"""
        if self.embed_dim != 0:
            emb_w = self.embedding(torch.LongTensor([word_id])) 
            print(emb_w)
    
    def save_embeddings(self, output_file):
        """Saves the embeddings of the model in the file given as input"""       
        if self.embed_dim != 0:
            print(tensor_desc(self.embedding.weight.data))
            emb_np = self.embedding.weight.detach().numpy()
            np.savetxt(output_file, emb_np)


def tensor_desc(x):
    """ Inspects a tensor: prints its type, shape and content"""
    print("Type:   {}".format(x.type()))
    print("Size:   {}".format(x.size()))
    print("Values: {}".format(x))
# ----------------------------------------------------------




# ----------------------------------------------------------
# Functions for data preparation
def get_index(word, word2idx, freeze=False):
    """
    map words to indices
    keep special OOV token (_UNK) at position 0
    """
    if word in word2idx:
        return word2idx[word]
    else:
        if not freeze:
            word2idx[word]=len(word2idx) #new index
            return word2idx[word]
        else:
            return word2idx["_UNK"]


def convert_to_n_hot(X, vocab_size):
    out = []
    for instance in X:
        n_hot = np.zeros(vocab_size)
        for w_idx in instance:
            n_hot[w_idx] = 1
        out.append(n_hot)
    return np.array(out)


def convert_to_one_hot(Y, label2idx, label_size):
    out = []
    for instance in Y:
        one_hot = np.zeros(label_size, dtype=int)
        one_hot[label2idx[instance]] = 1
        out.append(one_hot)
    return np.array(out)


# Format required by PyTorch's cross-entropy loss
def convert_to_index(Y, label2idx, label_size):
    out = []
    for instance in Y:
        index = label2idx[instance]
        out.append(index)
    return np.array(out)


# Format required by PyTorch's nn.embedding
def convert_to_indices(X):
    out = []
    for instance in X:
        indices = np.zeros(90, dtype=np.int)
        indices[:len(instance)] = instance
        indices[len(instance):] = 0
        out.append(indices)
    return np.array(out)
    
def join_function(sentence):
    return ' '.join(sentence)

def wordpiece_tokenizer(dataset):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized = []
    for data in dataset:
        tokenized.append(tokenizer.tokenize(data))
    return tokenized

def load_data(embed_dim):
    ### load data
    test = pd.read_csv('data/test.txt',sep=";",header=None)
    train = pd.read_csv('data/train.txt',sep=";",header=None)
    val = pd.read_csv('data/val.txt',sep=";",header=None)

    #Set random seed for replication purposes
    np.random.seed(500)

    # Splitting X and Y 
    X_train = train.iloc[:,0] 
    Y_train = train.iloc[:,1] 

    X_test = test.iloc[:,0] 
    Y_test = test.iloc[:,1] 

    X_val = test.iloc[:,0] 
    Y_val = test.iloc[:,1] 

    

    X_train = wordpiece_tokenizer(X_train)
    X_train = list(map(join_function,X_train))
    X_val = wordpiece_tokenizer(X_val)
    X_val = list(map(join_function,X_val))
    X_test = wordpiece_tokenizer(X_test)
    X_test = list(map(join_function,X_test))

    X_train,Y_train = np.array(X_train),np.array(Y_train)
    X_test,T_test = np.array(X_test),np.array(Y_test)
    X_val,Y_val = np.array(X_val),np.array(Y_val)

    Encoder = LabelEncoder()
    Y_train = Encoder.fit_transform(Y_train)
    Y_val   = Encoder.transform(Y_val)
    Y_test  = Encoder.transform(Y_test)

    ### create mapping word to indices
    word2idx = {"_UNK": 0}  # reserve 0 for OOV

    ### convert training etc data to indices
    X_train = [[get_index(w,word2idx) for w in x.split()] for x in X_train]
    freeze=True
    X_dev = [[get_index(w,word2idx,freeze) for w in x.split()] for x in X_val]
    X_test = [[get_index(w,word2idx,freeze) for w in x.split()] for x in X_test]
	
    #print("after word2idx {}".format(X_train[0]))

    vocab_size = len(word2idx)
    print("#vocabulary size: {}".format(len(word2idx)))
          
    if embed_dim == 0:
        X_train = convert_to_n_hot(X_train, vocab_size)
        X_dev = convert_to_n_hot(X_dev, vocab_size)
        X_test = convert_to_n_hot(X_test, vocab_size)
    else:
        X_train = convert_to_indices(X_train)
        X_dev = convert_to_indices(X_dev)
        X_test = convert_to_indices(X_test)

    #print("after conversion {}".format(X_train[0]))
    
    ### convert labels to one-hot
    label2idx = {label: i for i, label in enumerate(set(Y_train))}
    num_labels = len(label2idx.keys())
    print("#Categories: {}, {}".format(label2idx.keys(), label2idx.values()))
    y_train = convert_to_index(Y_train, label2idx, num_labels)
    y_dev = convert_to_index(Y_val, label2idx, num_labels)
    y_test = convert_to_index(Y_test, label2idx, num_labels)

    return X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx, label2idx


def load_animacy_sentences_and_labels(datafile):
    """
    loads the data set
    """
    input = [line.strip().split("\t") for line in open(datafile)]
    sentences = [sentence.split() for sentence, label in input]
    labels = [label for sentence, label in input]
    return sentences, labels
# ----------------------------------------------------------
    



## read input data
print("load data..")
X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx, tag2idx = load_data(100)
# Optional. It would be better (but would lead to considerably more complex code) 
# to load data on demand, using PyTorch's Dataset and DataLoader. See for example:
# https://github.com/joosthub/PyTorchNLPBook/blob/master/chapters/chapter_3/3_5_Classifying_Yelp_Review_Sentiment.ipynb
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader



print("#train instances: {}\n#dev instances: {}\n#test instances: {}".format(len(X_train),len(X_dev), len(X_test)))
assert(len(X_train)==len(y_train))
assert(len(X_test)==len(y_test))
assert(len(X_dev)==len(y_dev))

vocabulary_size=len(word2idx.keys())
num_classes = len(tag2idx)
input_size = len(X_train[0])
print("#Input size: {}".format(input_size))

print("#build model")
model = NN(input_dim=vocabulary_size, output_dim=num_classes)
model.to(dev)
#model.get_embedding(1) # get the embedding for the 2nd word in the vocab (the 1st one is UNK)
print("#Model: {}".format(model))
criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters())




print("#Training..")
num_epochs = 20
size_batch = 100
num_batches = len(X_train) // size_batch
print("#Batch size: {}, num batches: {}".format(size_batch, num_batches))
for epoch in range(num_epochs):
    # Optional. Shuffle the data (X and y) so that it is processed in different order in each epoch
    epoch_loss = 0
    for batch in range(num_batches):
        batch_begin = batch*size_batch
        batch_end = (batch+1)*(size_batch)
        X_data = X_train[batch_begin:batch_end]
        y_data = y_train[batch_begin:batch_end]
        

        X_tensor = torch.tensor(X_data, dtype=torch.int64)
        y_tensor = torch.tensor(y_data, dtype=torch.int64)
        X_tensor = X_tensor.to(dev) 
        y_tensor = y_tensor.to(dev) 

        optimizer.zero_grad()
        
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)

        loss.backward()
        optimizer.step()

        
        epoch_loss += loss.item()
        
    print("  End epoch {}. Average loss {}".format(epoch, epoch_loss/num_batches))
    

    print("  Validation")
    epoch_acc = 0
    num_batches_dev = len(X_dev) // size_batch
    print("    #num batches dev: {}".format(num_batches_dev))
    for batch in range(num_batches_dev):
        batch_dev_begin = batch*size_batch
        batch_dev_end = (batch+1)*(size_batch)
        X_data_dev = X_dev[batch_dev_begin:batch_dev_end]
        y_data_dev = y_dev[batch_dev_begin:batch_dev_end]

        X_tensor_dev = torch.tensor(X_data_dev, dtype=torch.int64)
        y_tensor_dev = torch.tensor(y_data_dev, dtype=torch.int64)
        X_tensor_dev = X_tensor_dev.to(dev) 
        y_tensor_dev = y_tensor_dev.to(dev) 
        
        y_pred_dev = model(X_tensor_dev)
        
        
        output = torch.max(y_pred_dev, dim=1)

        epoch_acc += accuracy_score(y_data_dev,output[1].cpu().data.numpy())

    print("    {}".format(epoch_acc / num_batches_dev))

epoch_acc = 0
num_batches_dev = len(X_test) // size_batch
print("    #num batches dev: {}".format(num_batches_dev))
for batch in range(num_batches_dev):
    batch_test_begin = batch*size_batch
    batch_test_end = (batch+1)*(size_batch)
    X_data_test = X_test[batch_test_begin:batch_dev_end]
    y_data_test = y_test[batch_test_begin:batch_dev_end]

    X_tensor_test = torch.tensor(X_data_test, dtype=torch.int64)
    y_tensor_test = torch.tensor(y_data_test, dtype=torch.int64)
    X_tensor_test = X_tensor_test.to(dev) 
    y_tensor_test = y_tensor_test.to(dev) 
    
    y_pred_test = model(X_tensor_test)
    
    
    output = torch.max(y_pred_test, dim=1)

    epoch_acc += accuracy_score(y_data_test,output[1].cpu().data.numpy())


print("Test accuracy    {}".format(epoch_acc / num_batches_dev))