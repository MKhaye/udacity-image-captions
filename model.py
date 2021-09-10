import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = embed_size
        self.hidden_size = hidden_size
        self.hidden = None
        self.embedding = nn.Embedding(vocab_size, self.input_size).cuda()
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,self.num_layers)
        self.linear =  nn.Linear(self.hidden_size, vocab_size).cuda() # not sure is the output should be the embedd size or the vocab size
        #word_scores = F.log_softmax(vocab_size)#dim=1

    
    def forward(self, features, captions,hidden):
        """
        args:
        features: containing the embedded image features
        captions: a PyTorch tensor corresponding to the last batch of captions 
                  all captions have the same length (captions.shape[1]), this is a 13 captions tensor
        
        return:
            outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size].
            Your output should be designed such that outputs[i,j,k] contains the model's predicted score,
            indicating how likely the j-th token in the i-th caption in the batch is the k-th token in the vocabulary.
        """
        #print("captions size ", captions.shape)
        #torch.cuda.empty_cache()

#         self.hidden = (torch.zeros(self.num_layers,captions.shape[0],self.hidden_size).cuda(),\
#                        torch.zeros(self.num_layers,captions.shape[0],self.hidden_size).cuda())
        #print("hidden size ", self.hidden_size)
#         if self.hidden == None:
#             self.hidden = self.init_hidden(captions.shape[0])#batch size
        #print("hidden size ", self.hidden_size)
        embeds = self.embedding(captions)
        #print("captions shape[1]", captions.shape[1])
        #print(" embedds     size ",embeds.transpose(1, 0)[:-1].shape)
        #print(" features size", features.shape)
        #ct = torch.cat((features.unsqueeze(0), embeds.transpose(1, 0))).transpose(1,0)
        #print("cat transposed done, sie of cat is : ",ct.transpose(1,0).shape)
        #print("cat done, sie of cat is : ",ct.shape)

        
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        #unsqueeze to add a new dimention to the feature for concatenation, 
        #transpose[-1]: for deleting the last element of the captions and replace it with the feature from the deocder

        x,hidden = self.lstm(torch.cat((features.unsqueeze(0), embeds.transpose(1, 0)[:-1])), hidden)
        
        #print("lstm out ", x.shape)
        #self.word_output(lstm_out.view(len(captions), -1))
        # get the scores for the most likely tag for a word
        x = x.transpose(1, 0)#this is mandatory to keep the same order and have batch first, making batch first on the definition fo the LSTM makes the hidden not compatible to made this one as a workarround
        #print("lstm out transpose", x.shape)
        x = x.reshape(x.size()[1]*x.size()[0], self.hidden_size)
        #print("x view shape", x.shape)
        x = self.linear(x)
        #print("outputs shape", x.shape)
        x = F.log_softmax(x,dim=1)
 
        #print("x post softmax ", x.shape)
        x = x.reshape(captions.size()[0],captions.size()[1],-1)
        #print("tag_outputs shape to return", x.shape)
        return x,hidden
    
    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, n_seqs, self.hidden_size).zero_().cuda(),
                weight.new(self.num_layers, n_seqs, self.hidden_size).zero_().cuda())
    
    
    def sample(self, inputs, states=None, max_len=20):
        """ input is a pytorsh tensor with the features of one single image,
        accepts pre-processed image tensor (inputs) and returns predicted 
        sentence (list of tensor ids of length max_len) """
        #encoder = EncoderCNN(embed_size)
        hidden = self.init_hidden(1)
        image_captions_init = inputs.data.new(self.num_layers, 1, self.hidden_size).zero_().cuda()
        image_captions_init = torch.tensor(image_captions_init).to(torch.int64)
        x,_ = self.forward(inputs,image_captions_init,hidden)
        print("ret : ", x)
        print(" shape : ", x.shape)

        return ''.join(x)
    
    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    
    def detach_elmt(self, h):
        if isinstance(h, tuple):
            return tuple(self.detach_elmt(v) for v in h)
        else:
            h.detach_()
            return Variable(h.detach())
