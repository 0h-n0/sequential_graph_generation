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
        self.gate_layer = torch.nn.Linear(hidden_dim, hidden_dim*2)
        self.embed_init_layer = torch.nn.Linear(hidden_dim + hidden_dim*2, hidden_dim)
        
        

    def propagate(self):
        pass

    def update_edges(self):
        pass

    def update_nodes(self):
        pass

        
    def forward(self, x, adj):
        # B, F, N = x.size()
        h = self.embed_node(x)
        # B, F, N, E = h.size()        
        h_v = self.initialize_embeded_node(h)
        return x

    def initialize_embeded_node(self, h):
        B, N, C = h.size()
        h_v = torch.randn(h.size()).view(-1, C)
        h_v_g = self.node_representation_layer(h_v)
        g_v = F.sigmoid(self.gate_layer(h_v)).view(-1, 1)
        print(g_v.size())
        print(h_v_g.size())
        h_G = torch.sum(torch.mul(g_v, h_v_g).view(B, N, -1), 1, keepdim=True)
        print(h_G.size())
        print(h.size())        
        _h = torch.cat((h, h_G), dim=2)
        print(_h.size())        
        _h_new = self.embed_init_layer(_h.view(-1, C))

        return _h._new(B, 1, -1)
                                       


if __name__ == '__main__':
    
    g = GraphGenerationProcess(5, 10, 10)
    x = torch.Tensor([[1, 0, 0, 0]]).long()
    adj = torch.eye(3).view(1, 3, 3)
    y = g(x, adj)
    
