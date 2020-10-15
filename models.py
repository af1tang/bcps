#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonymous
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import utils

##### Main T-S Models #####
class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        #recurrent gates
        self.W_xr = torch.nn.parameter.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = torch.nn.parameter.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.br =torch.nn.parameter.Parameter(torch.Tensor(hidden_size))
        self.W_xr = torch.nn.init.xavier_uniform_(self.W_xr)
        self.W_hr = torch.nn.init.xavier_uniform_(self.W_hr)
        self.br = torch.nn.init.zeros_(self.br)
        #forget gates
        self.W_xz = torch.nn.parameter.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = torch.nn.parameter.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bz = torch.nn.parameter.Parameter(torch.Tensor(hidden_size))
        self.W_xz = torch.nn.init.xavier_uniform_(self.W_xz)
        self.W_hz = torch.nn.init.xavier_uniform_(self.W_hz)
        self.bz = torch.nn.init.zeros_(self.bz)
        #input gates
        self.W_xg = torch.nn.parameter.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = torch.nn.parameter.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg =torch.nn.parameter.Parameter(torch.Tensor(hidden_size))
        self.W_xg = torch.nn.init.xavier_uniform_(self.W_xg)
        self.W_hg = torch.nn.init.xavier_uniform_(self.W_hg)
        self.bg = torch.nn.init.zeros_(self.bg)

    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """
        z = F.sigmoid(torch.mm(x, self.W_xz) + torch.mm(h_prev, self.W_hz) +self.bz) 
        r = F.sigmoid(torch.mm(x, self.W_xr) + torch.mm(h_prev, self.W_hr) +self.br) 
        g = F.tanh(torch.mm(x, self.W_xg) + r*(torch.mm(h_prev, self.W_hg) +self.bg))
        h_new = (1-z)*g + z*h_prev
        return h_new
    
class InterpGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InterpGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        #recurrent gates
        self.W_xr = torch.nn.parameter.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = torch.nn.parameter.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.br =torch.nn.parameter.Parameter(torch.Tensor(hidden_size))
        self.W_xr = torch.nn.init.xavier_uniform_(self.W_xr)
        self.W_hr = torch.nn.init.xavier_uniform_(self.W_hr)
        self.br = torch.nn.init.zeros_(self.br)
        #forget gates
        self.W_xz = torch.nn.parameter.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = torch.nn.parameter.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bz = torch.nn.parameter.Parameter(torch.Tensor(hidden_size))
        self.W_xz = torch.nn.init.xavier_uniform_(self.W_xz)
        self.W_hz = torch.nn.init.xavier_uniform_(self.W_hz)
        self.bz = torch.nn.init.zeros_(self.bz)
        #input gates
        self.W_xg = torch.nn.parameter.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = torch.nn.parameter.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg =torch.nn.parameter.Parameter(torch.Tensor(hidden_size))
        self.W_xg = torch.nn.init.xavier_uniform_(self.W_xg)
        self.W_hg = torch.nn.init.xavier_uniform_(self.W_hg)
        self.bg = torch.nn.init.zeros_(self.bg)

    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """
        z = F.sigmoid(torch.mm(x, self.W_xz) + torch.mm(h_prev, self.W_hz) +self.bz) 
        r = F.sigmoid(torch.mm(x, self.W_xr) + torch.mm(h_prev, self.W_hr) +self.br) 
        g = F.tanh(torch.mm(x, self.W_xg) + r*(torch.mm(h_prev, self.W_hg) +self.bg))
        h_new = (1-z)*g + z*h_prev
        return h_new, g, z, r
    
class DualFunction(nn.Module):
    def __init__(self, budget):
        super(DualFunction, self).__init__()
        self.lambda_reg = nn.Parameter(torch.FloatTensor([.001]), requires_grad = True)
        self.budget = budget

    def forward(self, masks, dtype=torch.cuda.FloatTensor):
        '''
        masks: outputs of G
        budget: constraint param
        '''
        cost = self.lambda_reg * (torch.sum(torch.cat([torch.norm(A,1).view(-1) for A in masks])) - self.budget)
        constraint = torch.clamp(cost, 0., 1000.).squeeze()
        return constraint
    
class Generator(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Generator, self).__init__()

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        #self.budget = budget
        
        self.threshold = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

        self.out = nn.Linear(hidden_size, feature_size)
        self.gru = MyGRUCell(feature_size, hidden_size)
        #self.dual = DualFunction()

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        hidden = self.init_hidden(batch_size)

        #encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        masks = []; mask = torch.ones(batch_size, self.feature_size)
        masks.append(mask)
        for i in range(seq_len-1):
            x = inputs[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru(x*mask, hidden)
            mask = F.sigmoid(self.out(hidden).squeeze())
            #fc = F.threshold(self.linear(hidden), threshold=1., value = 0.)
            #mask = F.hardshrink( self.out(fc).squeeze(), lambd=1.)
            #mask = torch.where(mask > self.threshold, torch.ones(1), mask)
            masks.append(mask)

        masks = torch.stack(masks, dim=1)
        #outputs = masks * inputs
        
        #constraint = self.dual(masks, self.budget)
        return masks

    def init_hidden(self, batch_size):
        return utils.to_var(torch.zeros(batch_size, self.hidden_size))

class Discriminator(nn.Module):
    def __init__(self, feature_size,hidden_size, output_size, num_layers = 2,
                 dropout = 0.2, bidirectional = False):
        super(Discriminator, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        
        #self.rnn = nn.GRU(input_size =feature_size, hidden_size = hidden_size, 
        #                  num_layers = num_layers, bidirectional = self.bidirectional,
        #                  batch_first = True, dropout = dropout)
        #self.out = nn.Linear(hidden_size, output_size)
        self.rnn = MyGRUCell(input_size=feature_size, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        self.hidden = self.init_hidden(batch_size)
        #rnn_out, self.hidden = self.rnn(
        #        x.view(batch_size, seq_len, self.feature_size), self.hidden)
        #output = self.out(rnn_out[:,-1,:]).squeeze()
        #output = F.softmax(output)
        annotations = []
        for i in range(seq_len):
            x = inputs[:,i,:]  # Get the current time step, across the whole batch
            self.hidden = self.rnn(x, self.hidden)
            annotations.append(self.hidden)
        annotations = torch.stack(annotations, dim=1)
        attention_weights = self.attention(self.hidden, annotations)
        context = torch.sum(attention_weights* annotations, dim=1)
        output = self.out(context)
        output = F.softmax(output)
        return output
    
    def init_hidden(self, batch_size):
        #if self.bidirectional == True:
        #    return torch.autograd.Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_size))
        #else:
        #    return torch.autograd.Variable(torch.zeros(1*self.num_layers, batch_size, self.hidden_size)) 
        return utils.to_var(torch.zeros(batch_size, self.hidden_size))
    
class InterpGRU(nn.Module):
    def __init__(self, feature_size,hidden_size, output_size):
        super(InterpGRU, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = MyGRUCell(input_size=feature_size, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        self.hidden = self.init_hidden(batch_size)
        #annotations, gs, rs, zs = [], [], [], []
        for i in range(seq_len):
            x = inputs[:,i,:]  # Get the current time step, across the whole batch
            self.hidden = self.rnn(x, self.hidden)
            #annotations.append(self.hidden)
            #gs.append(g); rs.append(r); zs.append(z)
        #annotations = torch.stack(annotations, dim=1)
        #gs = torch.stack(gs, dim=1); rs = torch.stack(rs, dim=1); zs = torch.stack(zs,dim=1)
        #attention_weights = self.attention(self.hidden, annotations)
        #context = torch.sum(attention_weights* annotations, dim=1)
        #output = self.out(context)
        output = self.out(self.hidden)
        output = F.softmax(output)
        return output#, gs, rs, zs
    
    def init_hidden(self, batch_size):
        return utils.to_var(torch.zeros(batch_size, self.hidden_size))   
    
#### Baseline Models ####
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        # Create a two layer fully-connected network. Hint: Use nn.Sequential
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1

        self.attention_network = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, annotations):
        """The forward pass of the attention mechanism.

        Arguments:
            hidden: The current decoder hidden state. (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            output: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size, seq_len, hid_size = annotations.size()
        expanded_hidden = hidden.unsqueeze(1).expand_as(annotations)

        concat = torch.cat([expanded_hidden, annotations], dim=2)
        reshaped_for_attention_net = concat.reshape((batch_size, seq_len, hid_size*2))
        attention_net_output = self.attention_network(reshaped_for_attention_net)
        unnormalized_attention = attention_net_output.reshape((batch_size, seq_len, 1))  # Reshape attention net output to have dimension batch_size x seq_len x 1

        return self.softmax(unnormalized_attention)


class AttentionGRU(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super(AttentionGRU, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.rnn = MyGRUCell(input_size=feature_size, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        self.hidden = self.init_hidden(batch_size)
        annotations = []
        for i in range(seq_len):
            x = inputs[:,i,:]  # Get the current time step, across the whole batch
            self.hidden = self.rnn(x, self.hidden)
            annotations.append(self.hidden)
        annotations = torch.stack(annotations, dim=1)
        attention_weights = self.attention(self.hidden, annotations)
        context = torch.sum(attention_weights* annotations, dim=1)
        output = self.out(context)
        output = F.sigmoid(output)
        return output, attention_weights
    
    def init_hidden(self, batch_size):
        return utils.to_var(torch.zeros(batch_size, self.hidden_size))

class SelfAttention(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.rnn = MyGRUCell(input_size=feature_size, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=feature_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        self.hidden = self.init_hidden(batch_size)
        activations = []
        for i in range(seq_len):
            x = inputs[:,i,:]  # Get the current time step, across the whole batch
            attention_weights = self.attention(x, inputs)
            context = torch.sum(attention_weights* inputs, dim=1)
            activations.append(context)
            self.hidden = self.rnn(context, self.hidden)
        output = self.out(self.hidden)
        output = F.sigmoid(output)
        return output, activations
    
    def init_hidden(self, batch_size):
        return utils.to_var(torch.zeros(batch_size, self.hidden_size))

class MLP(nn.Module):
    def __init__(self, feature_size, output_size, hidden_size, dropout=.3):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(feature_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)        
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        h = self.relu(self.linear1(x))
        h = self.dropout(self.relu(self.linear2(h)))
        h = self.dropout(self.relu(self.linear2(h)))
        output = F.sigmoid(self.linear3(h))
        return output


class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size,
                 output_size, bidirectional = False, num_layers = 2,
                 dropout = .0, return_sequence = False):
        super(GRU, self).__init__()  
        self.feature_size = feature_size
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size =feature_size, hidden_size = hidden_size, 
                          num_layers = num_layers, bidirectional = bidirectional, dropout= dropout,
                          batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        self.hidden = self.init_hidden(batch_size)
        gru_out, self.hidden = self.gru(
                x.view(batch_size, seq_len, self.feature_size), self.hidden)
        output = self.linear(gru_out[:, -1, :]).squeeze()
        output = F.sigmoid(output)
        return output
    
    def init_hidden(self, batch_size):
        if self.bidirectional == True:
            return torch.autograd.Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_size))
        else:
            return torch.autograd.Variable(torch.zeros(1*self.num_layers, batch_size, self.hidden_size)) 
