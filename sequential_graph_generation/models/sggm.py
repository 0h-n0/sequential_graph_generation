import math

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from .basic import GraphLinear


class PropagationBlock(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 celltype="GRU",
                 reccurent_dropout=0.5,
                 reverse_direction=False,
                 num_round=3,
                 ):
        super(PropagationBlock, self).__init__()
        self.hidden_size = hidden_size
        self.reccurent_dropout = reccurent_dropout
        self.reverse_direction = reverse_direction
        self.num_round = num_round
        self.rnn_cells = nn.ModuleList([
            getattr(nn, celltype)(2 * 3 * hidden_dim, 3, 3)
            for i in range(num_round)])
        
        self.foward_message_layer = \
                            nn.ModuleList([
                            nn.Linears(3 * hidden_dim, 2 * 3 * hidden_dim)
                            for i in range(num_round)])
        
        if self.reverse_direction:
            self.reverse_message_layer = \
                            nn.ModuleList([                                         
                            nn.Linears(3 * hidden_dim, 2 * 3 * hidden_dim)
                            for i in range(num_round)])                                                                
        # Multiplying by 2 in the number of output dimenssion is hyper parameter.
        # And this value comes from original paper.

    def forward(self, embedded_node, embedded_adjancy_matrix):
        N = pairlist.size(0)
        message_inputs = []        
        _message_inputs = []
        pre_b = None
        batchsize = embedded_node.size(0)
        sequence_lenghts = embedded_node.size(1)
        
        for T in range(self.num_round):
            for B in range(batchsize):
                for S in range(sequence_lenghts):
                    message_inputs.append(torch.stack(_message_inputs))
                    _message_inputs = []
                    _message_inputs.append(torch.cat((h_node[b, :, :],
                                                      h_node[b, :, :],
                                                      h_edge[b, i, j, :])))
                    _message_inputs.append(torch.cat((h_node[b, i, :],
                                                  h_node[b, j, :],
                                                  h_edge[b, i, j, :])))
                if n == N-1:
                    message_inputs.append(torch.stack(_message_inputs))                    
            pre_b = b

        m = rnn_utils.pad_sequence(message_inputs, batch_first=True)

        B, NN, C = m.size()     # batch, number of node type, channel
        m = m.view(B*NN, C)
        m = self.foward_message_layer(m)
        m = o.view(B, NN, C)
        ## TODO: add aggrigation layer
        ## TODO: add update layer
        return h_G


class SGGM(torch.nn.Module):
    '''
    SGGM stands for 'Sequential Generative Graph Model'.
    * see: https://arxiv.org/abs/1803.03324
    '''
    def __init__(self,
                 batchsize,
                 hidden_dim:int,
                 num_node_type: int,
                 num_edge_type: int,
                 max_num_node: int=15,           
                 num_pair_type: int=None,
                 celltype="GRU"):
        super(SGGM, self).__init__()
        torch.manual_seed(1)

        self.batchsize = batchsize
        self.hidden_dim = hidden_dim
        self.num_node_type = num_node_type        # NN
        self.num_edge_type = num_edge_type        # NE
        self.max_num_node = max_num_node
        
        if num_pair_type == None:
            self.num_pair_type = num_pair_type
        else:
            # calculate permutation
            self.num_pair_type = math.factorial(num_node_type) \
                                 // math.factorial(num_node_type - 2) \
                                 * num_edge_type
            
        self.embed_node_layer = torch.nn.Embedding(num_node_type, hidden_dim, padding_idx=0)
        self.embed_node_type_layer = torch.nn.Embedding(num_node_type, hidden_dim, padding_idx=0)
        self.embed_edge_layer = torch.nn.Embedding(num_edge_type, hidden_dim, padding_idx=0)
        

        if num_node_type == 1:
            self.addnode_layer = nn.Linear(num_node_type, num_node_type)
        else:
            self.addnode_layer = nn.Linear(num_node_type, num_node_type + 1)

        
        # self.reverse_message_layer = nn.Linear(3*hidden_dim, 2*3*hidden_dim)
        # Not use reverse layer. This is because I don't support parsing task with this model.

    def adj_matrix_to_nodelist(self, adj_matrix):
        return torch.nonzero(adj_matrix)

    def get_edge_features(self, adj_matrix):
        # Not use.
        # Edge features must be positive integer which represents a type of edge.        
        sequence_lenghts = []
        B = adj_matrix.size(0)
        features = []
        for i in range(B):
            mask = adj_matrix[i].ge(1)        
            features.append(torch.masked_select(adj_matrix[i], mask))
        return rnn_utils.pad_sequence(features, batch_first=True)
    
    def forward(self):
        """x
        Generate distribution from batchsize.
        """
        completed_graph_list = []

        vertices = torch.tensor([]).long()
        edges = torch.tensor([]).long()
        
        samples = self.f_addnode()
        while True:
            vertices = torch.cat((vertices, samples))
            add_edge_flag = self.f_addedge()
            h_nodes = self.get_node_reporesentation(h_nodes,)
            while add_edge_flag == 1:
                self.f_nodes()
                add_edge_flag = self.f_addedge()
            samples = self.f_addnode()
                
    def f_nodes(self):
        pass
            
    def f_addedge(self):
        pass
            
    def f_addnode(self, h_G=None):
        if h_G is None:
            h_G = torch.zeros(self.batchsize, self.num_node_type)
        _dist = dists.Categorical(logits=F.log_softmax(self.addnode_layer(h_G)))            
        samples = _dist.sample().view(self.batchsize, 1)
        return samples
            
    def initilize_h_node(self, h_nodes):
        h_edge = self.embed_edge_layer(adj_matrix)        
        h_G = self.propagate(h_node, h_edge, pair_list)

    def aggrigate(self, m, parlist):
        pass

if __name__ == '__main__':
    g = SGGM(1, 5, 11, 10)
    #y = g()

    x = torch.Tensor([[1, 10, 4, 3, 1, 3], [1, 10, 4, 3, 0, 0]]).long()
    adj_1 = torch.nn.ZeroPad2d((1, 0, 0, 1))(torch.eye(5))
    adj_2 = torch.nn.ZeroPad2d((1, 1, 0, 2))(torch.eye(4))
    adj_1 += adj_1.t()
    adj_2 += adj_2.t()    
    padded_adj_matrix = torch.stack((adj_1, adj_2)).long()

    b = PropagationBlock(10)
    
