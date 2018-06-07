import math

import torch
import torch.nn.functional as F

class GraphGenerationProcess(torch.nn.Module):
    def __init__(self,
                 hidden_dim:int,
                 num_node_type: int,
                 num_edge_type: int,
                 num_pair_type: int=None,                 
                 celltype="GRU"):
        super(GraphGenerationProcess, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        if num_pair_type == None:
            self.num_pair_type = num_pair_type
        else:
            # calculate permutation
            self.num_pair_type = math.factorial(num_node_type) \
                                 // math.factorial(num_node_type - 2) \
                                 * num_edge_type
        self.celltype = celltype        
        self.embed_node = torch.nn.Embedding(num_node_type, hidden_dim, padding_idx=0)
        
        #hv initialize
        self.node_representation_layer = torch.nn.Linear(hidden_dim, hidden_dim*2)
        self.graph_representation_layer = torch.nn.Linear(hidden_dim, 1)
        self.f_init = torch.nn.Linear(hidden_dim + hidden_dim*2, hidden_dim)
        
        

    def propagate(self):
        pass

    def update_edges(self):
        pass

    def update_nodes(self):
        pass

        
    def forward(self, x, adj):
        h = self.embed_node(x)
        h_v = self.initialize_embeded_node(h)
        return x

    def initialize_embeded_node(self, h):

        B, F, H = h.size()
        h_v = torch.randn(h.size()).view(-1, H)
        print(h_v)
        





if __name__ == '__main__':
    
    g = GraphGenerationProcess(5, 3, 3)
    x = torch.Tensor([[[0, 0, 1]]]).long()
    adj = torch.eye(3).view(1, 3, 3)
    y = g(x, adj)
    
