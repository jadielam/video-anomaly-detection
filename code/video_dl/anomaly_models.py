from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.utils.model_zoo
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

class Classifier(nn.Module):
    '''
    Standard multilayer perceptron classifier.
    '''
    def __init__(self, nb_classes, input_size, nb_layers = 3,
                classifier_function = partial(F.log_softmax, dim = 1),
                activation_function = F.relu, 
                hidden_dimension = 1024,
                dropout = 0.2):
        '''
        Arguments:
        - nb_classes (int): number of classes for classification
        - input_size (int): length of the input vector
        - nb_layers (int): number of layers for the classifier
        - classifier_function: (:obj:`torch.nn.functional`) A function to apply to the last layer.
                             Usually will be softmax or sigmoid
        '''
        super(Classifier, self).__init__()
        self.classifier_function = classifier_function
        self.activation_function = activation_function

        dims = []
        for i in range(max(nb_layers, 1)):
            dim_entry = [hidden_dimension, hidden_dimension]
            if i == 0: 
                dim_entry[0] = input_size
            if i == max(nb_layers, 1) - 1:
                dim_entry[1] = nb_classes
            dims.append(dim_entry)

        self.dropout_layer = nn.Dropout(p = dropout)
        self.layers = []
        for i in range(len(dims)):
            linear_layer = nn.Linear(dims[i][0], dims[i][1])
            self.layers.append(linear_layer)
            super(Classifier, self).add_module("linear_{}".format(i), linear_layer)

    def forward(self, input_t):
        '''
        Arguments:
        - input_t_l (:obj:`torch.Tensor`): input tensor to use for classification of size (batch, input_size)

        Returns:
        - output (:obj:`torch.Tensor`) of size (batch, nb_classes)
        '''
        next_t = input_t
        next_t = self.dropout_layer(next_t)
        for i in range(len(self.layers) - 1):
            next_t = self.activation_function(self.layers[i](next_t))
        next_t = self.layers[len(self.layers) - 1](next_t)
        output = self.classifier_function(next_t)
        return output

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input_t):
        return input_t

class AnomalyModel(nn.Module):

    def __init__(self, gru_dropout):
        super(AnomalyModel, self).__init__()
        #1. Get a vgg16 or something model from torchvision
        self._output_dim = 2048
        self._hidden_size = 100
        self._gru_dropout = gru_dropout

        self._vision_features = torchvision.models.resnet50(pretrained = True)
        self._vision_features.fc = Identity()
        
        #TODO: FIgure out the size of the output of vision_features
        #4. Create RNN module
        self._gru_cell = nn.GRUCell(self._output_dim, self._hidden_size, bias = True)
        
        self._classifier = Classifier(2, self._hidden_size)
        self._criterion = nn.NLLLoss()

        #5. Store previous hidden layer here.
        if use_cuda:
            self._previous_hidden = Variable(torch.zeros(1, self._hidden_size)).cuda()
        else:
            self._previous_hidden = Variable(torch.zeros(1, self._hidden_size))
        
    def loss(self, input_t, ground_truth):
        classification = self.forward(input_t)
        loss = self.criterion(classification, ground_truth)
        return loss
    
    def forward(self, input_t):
        '''
        Arguments:
        - input_t (:obj: `torch.Tensor`) of size (331, 331, 3), which is essentially a picture. 

        Returns:
        - classification (:obj: `torch.Tensor`) of size (nb_classes)
        '''
        input_t = input_t.unsqueeze(0)
        features = self._vision_features(input_t)
        new_hidden = self._gru_cell(features, self._previous_hidden)
        self._previous_hidden = new_hidden
        classification = self.classifier(new_hidden)
        classification = classification.squeeze()
        return classification
    
    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad,
            super(AnomalyModel, self).parameters())

    def use_cuda(self):
        return use_cuda

class BetterAnomalyModel(nn.Module):

    def __init__(self, gru_dropout, seq_len):
        super(BetterAnomalyModel, self).__init__()
        #1. Get a vgg16 or something model from torchvision
        self._output_dim = 2048   # THe dimension of the output by the resnet network.
        self._hidden_size = 350
        self._gru_dropout = gru_dropout
        
        self._vision_features = torchvision.models.resnet50(pretrained = True)

        self._vision_features.fc = Identity()
        
        #TODO: FIgure out the size of the output of vision_features
        #4. Create RNN module
        self._gru = nn.GRU(self._output_dim, self._hidden_size, bidirectional = False,
                            batch_first = False, dropout = self._gru_dropout)
        
        self._classifier = Classifier(2, self._hidden_size)
        self._criterion = nn.NLLLoss()

        #5. Store previous hidden layer here.
    
        if use_cuda:
            self._init_hidden = Variable(torch.zeros(self._gru.num_layers * 1, 1, self._hidden_size)).cuda()
        else:
            self._init_hidden = Variable(torch.zeros(self._gru.num_layers * 1, 1, self._hidden_size))
        
    def loss(self, input_t, ground_truth):
        classification = self.forward(input_t)
        loss = self._criterion(classification, ground_truth)
        return loss, classification
    
    def init_hidden(self):
        if use_cuda:
            hidden = Variable(torch.randn((self._gru.num_layers * 1, 1, self._hidden_size))).cuda()
        else:
            hidden = Variable(torch.randn((self._gru.num_layers * 1, 1, self._hidden_size)))
        return hidden

    def forward(self, input_t):
        '''
        Arguments:
        - input_t (:obj: `torch.Tensor`) of size (16, 3, 331, 331), which is essentially a picture. 

        Returns:
        - classification (:obj: `torch.Tensor`) of size (nb_classes)
        '''
        features = self._vision_features(input_t)  #output should be (1, output_dim)
        features = torch.unsqueeze(features, 1)
        _, hidden_out = self._gru(features, self.init_hidden())
        classification = self._classifier(hidden_out)
        classification = classification.view(1, -1)
        return classification
    
    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad,
            super(BetterAnomalyModel, self).parameters())

    def use_cuda(self):
        return use_cuda
