import math

import torch
import torch.nn.functional as F

class GraphGenerationProcess(torch.nn.Module):
    def __init__(self,
                 hidden_dim:int,
                 num_node_type: int,
                 num_edge_type: int,
                 num_pair_type: int=None,
                 max_num_node: int=15,
                 num_round: int=3,
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

        self.max_num_node = max_num_node
        self.num_round = num_round            
        self.celltype = celltype

        ## embedding
        self.embed_node_layer = torch.nn.Embedding(num_node_type, hidden_dim, padding_idx=0)
        
        ## hv initializtion layer
        self.node_representation_layer = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate_layer = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.embed_init_layer = torch.nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)

        ## propagation layer
        if celltype == "GRU":
            _cell = torch.nn.GRUCell
        elif celltype == "LSTM":
            _cell = torch.nn.LSTMCell

        input_dim = 3 * hidden_dim
        output_dim = 2 * input_dim
        # output dimension from init layer => 3 * hidden_dim
        # forward and backward message => 2

        self.edge_forward_message_layer = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim) for _ in range(self.num_round)])
        self.edge_reverse_message_layer = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim) for _ in range(self.num_round)])            
        
        self.propagation_cells = torch.nn.ModuleList([_cell(output_dim, hidden_dim)
                                                      for _ in range(self.num_round)])

        self.propagation_node_representation_layer = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.propagation_gate_layer = torch.nn.Linear(hidden_dim, 1)
        self.activation_layer = torch.nn.Linear(hidden_dim*2, 1+self.num_node_type)


    def generate(self, x):
        pass

    

    def forward(self, x):
        # B, F, N = x.size()
        node_index = x.view(-1, 1)        
        embed_node = self.embed_node_layer(node_index)
        num_node = embed_node.size(0)
        
        edge_index = torch.zeros(num_node, 1, self.max_num_node)
        embed_edge = torch.zeros(num_node, 1, self.max_num_node, self.hidden_dim)
        adj_mat = torch.zeros(num_node, 1, self.max_num_node)

        h_v = self.initialize_embeded_node(embed_node)
        h_G = self.propagate(h_v, embed_edge, adj_mat)
        p_addnode = F.softmax(self.activation_layer(h_G), dim=1)
        DC = torch.distributions.Categorical(p_addnode)
        v_new_index = DC.sample()

        node_index = torch.log(node_index.float())
        node_index = torch.cat((node_index,DC.log_prob(v_new_index).view(-1,1)),1)
        check_node = (v_new_index.view(-1,1) != 0).float()
        print(node_index.size())
        position = 1
        while torch.sum(check_node) != 0 and position < self.N_Max_atom -1:
            #print('atom position')
            #print(position)
            emb_new_node = self.E_atom(v_new_index.view(-1,1))
            v_new = self.h_v_init(h_v,emb_new_node,self.f_m_init,self.g_m_init,self.f_init)
            
            h_v = torch.cat((h_v,v_new),1)
            
            emb_edge = torch.cat((emb_edge, torch.zeros(emb_edge.size(0),1,emb_edge.size(2),emb_edge.size(3))),1)
            edge_index = torch.cat((edge_index, torch.zeros(len(h_v), 1, self.N_Max_atom)), 1)
            adj_mat = torch.cat((adj_mat, torch.zeros(adj_mat.size(0), 1, adj_mat.size(2))), 1)
            
            h_G = self.propagation(h_v,emb_edge,adj_mat,self.f_e_ae,self.f_n_ae,self.f_m_ae,self.g_m_ae)
            
            p_addedge = F.softmax(self.f_ae(h_G),dim=-1)
            DB = Bernoulli(p_addedge)
            z_t = DB.sample().view(-1,1)
            check_edge = torch.mul(z_t, check_node)            
            check_edge_s = torch.ones(h_v.size(0),h_v.size(1),1)
            limit = 15
            count = 0
            while torch.sum(check_edge) != 0 and torch.sum(check_edge_s) != 0 and count<limit:
                h_u_T = self.prop_only(h_v,emb_edge,adj_mat,self.f_e_s,self.f_n_s)
                h_v_T = torch.cat([h_u_T[:,-1,:].view(len(h_u_T),1,-1)for _ in range(h_u_T.size(1))],1)

                """
                
                p_nodes = F.softmax(self.f_s(torch.cat((h_u_T,h_v_T),2).view(-1,h_u_T.size(2)*2)),dim=1).view(h_u_T.size(0), h_u_T.size(1), -1)
                DC_edge = Categorical(p_nodes)
                edge_new_index = DC_edge.sample().view(h_u_T.size(0), h_u_T.size(1), -1).float()
                check_edge_s_cdt = (edge_new_index == 0).float().view(h_u_T.size(0), h_u_T.size(1), -1)
                edge_new_index =  torch.mul(edge_new_index, check_edge.view(-1, 1, 1))
                edge_new_index =  torch.mul(edge_new_index, check_edge_s)
                
                edge_index = edge_index.clone()
                
                edge_index[:,:,position] = edge_new_index.view(-1,edge_new_index.size(1))
                edge_index[:,position,:position+1] = edge_new_index.view(-1,edge_new_index.size(1))
                emb_edge = self.E_bond(edge_index.long())

                adj_mat = adj_mat.clone()
                adj_sub = edge_new_index != 0
                adj_mat[:,:,position] = adj_sub.long().view(-1,adj_sub.size(1))
                adj_mat[:,position,:position+1] = adj_sub.long().view(-1,adj_sub.size(1))
                

                h_G = self.propagation(h_v,emb_edge,adj_mat,self.f_e_ae,self.f_n_ae,self.f_m_ae,self.g_m_ae)

                p_addedge = F.softmax(self.f_ae(h_G),dim=-1)
                z_t = Bernoulli(p_addedge).sample().view(-1,1)
                check_edge = torch.mul(z_t, check_edge)
                check_edge_s = torch.mul(check_edge_s, check_edge_s_cdt)
                #print('check_edge')
                #print(torch.sum(check_edge))
                #print('check_edge_s')
                #print(torch.sum(check_edge_s))
                count +=1
                position += 1
                h_G = self.propagation(h_v,emb_edge,adj_mat,self.f_e_ae,self.f_n_ae,self.f_m_ae,self.g_m_ae)
                p_addnode = F.softmax(self.f_an(h_G),dim=-1)
                DC = Categorical(p_addnode)
                v_new_index = DC.sample()
                node_index = torch.cat((node_index,DC.log_prob(v_new_index).view(-1,1)),1)
                check_node = (v_new_index.view(-1,1) != 0).float()
                #print('sum of check_node')
                #print(torch.sum(check_node))
        if node_index.size(1) < self.N_Max_atom:
            adjust =torch.zeros(node_index.size(0),self.N_Max_atom - node_index.size(1))
            node_index = torch.cat((node_index.float(),adjust),1)
        else:
            pass
        return node_index.float(), edge_index
        """
        return x


    def propagate_only(self, h_v, embed_edge, adj_mat):
        pass
    
    def propagate(self, h_v, embed_edge, adj_mat):
        for T in range(self.num_round):
            for t in range(len(h_v[0])):
                if t == 0:
                    neighbor = torch.mul(h_v, adj_mat[:, t, :h_v.size(1)].view(-1, h_v.size(1), 1))
                    watch = torch.mul(h_v[:,t,:].view(-1,1,h_v.size(2)), adj_mat[:,t,:h_v.size(1)].view(-1, h_v.size(1), 1))
                    m_u_v = self.edge_forward_message_layer[T](torch.cat((neighbor,watch,embed_edge[:,t,:h_v.size(1)]),2).view(neighbor.size(0)*neighbor.size(1), -1))
                    m_u_v = torch.sum(m_u_v.view(neighbor.size(0), neighbor.size(1),-1),1,keepdim=True)
                    m_v_u = self.edge_reverse_message_layer[T](torch.cat((watch,neighbor,embed_edge[:,t,:h_v.size(1)]),2).view(neighbor.size(0)*neighbor.size(1), -1))
                    m_v_u = torch.sum(m_v_u.view(neighbor.size(0), neighbor.size(1), -1),1,keepdim=True)
                    a_v = m_u_v + m_v_u                    
                    pass
                else:
                    neighbor = torch.mul(h_v, adj[:,t,:h_v.size(1)].view(-1, h_v.size(1), 1))
                    pass
            h_v = self.propagation_cells[T](a_v.view(-1,a_v.size(2)), h_v.view(-1, h_v.size(2))).view(h_v.size(0), h_v.size(1), -1)
        h_v_g = self.propagation_node_representation_layer(h_v.view(h_v.size(0)* h_v.size(1), -1))
        g_v = F.sigmoid((self.propagation_gate_layer(h_v)).view(-1,1))
        h_G = torch.sum(torch.mul(g_v,h_v_g).view(h_v.size(0),h_v.size(1),-1),1,keepdim=False)
        return h_G
        

    def initialize_embeded_node(self, h):
        B, N, C = h.size()
        h_v = torch.randn(h.size()).view(-1, C)
        h_v_g = self.node_representation_layer(h_v).view(B, N, -1)
        g_v = F.sigmoid(self.gate_layer(h_v)).view(-1, C).view(B, N, -1)
        h_G = torch.sum(torch.mul(g_v, h_v_g).view(B, N, -1), 1, keepdim=True)
        _h = torch.cat((h, h_G), dim=2).view(-1, C+2*C)
        _h_new = self.embed_init_layer(_h).view(B, 1, -1)
        return _h_new


if __name__ == '__main__':
    g = GraphGenerationProcess(5, 10, 10)
    x = torch.Tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0]]).long()
    y = g(x)
    
