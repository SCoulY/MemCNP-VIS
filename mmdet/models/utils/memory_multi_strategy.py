import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryN2N(nn.Module):        
    def __init__(self, hdim, kdim, 
                 moving_average_rate=0.999):
        super().__init__()
        
        self.c = hdim
        self.k = kdim
        self.moving_average_rate = moving_average_rate
        self.feat_units = nn.Embedding(kdim, hdim)
        self.label_units = nn.Embedding(kdim, 4)
                
    def update(self, x, y, score):
        '''
            x: (n, c)
            y: (n, 4)
            e: (k, c)
            score: (n, k)
        '''
        with torch.no_grad():
            m1 = self.feat_units.weight.data
            m2 = self.label_units.weight.data
            
            x = x.detach()
            y = y.detach()
            embed_ind = torch.sort(score, dim=1, descending=True)[1] # (n, k)
            embed_bot = embed_ind[:, int(0.1*self.k):] # (n, 0.9k)
            batch_ind = torch.arange(x.size(0), device=x.device)[:,None]
            score[batch_ind, embed_bot] = 0 
            score = F.softmax(score, dim=1)

            embed_sum = x.transpose(0, 1) @ score # (c, k) 
            embed_mean = embed_sum / (score.sum(dim=0) + 1e-6) # (c, k)
            embed_label_sum = y.transpose(0, 1) @ score # (4, k) 
            embed_label_mean = embed_label_sum / (score.sum(dim=0) + 1e-6) # (4, k)
    
            new_feat_units = m1 * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
            new_label_units = m2 * self.moving_average_rate + embed_label_mean.t() * (1 - self.moving_average_rate)
            if self.training:
                self.feat_units.weight.data = new_feat_units
                self.label_units.weight.data = new_label_units
        return new_feat_units, new_label_units
                
    def forward(self, x, y=None, update_flag=None):
        '''
          x: feats of shape(b, c, h, w) 
          y: gt of shape (b, 4, h, w) with in-box value lrtb and out-box value 0
          embed: (k, c)
          output: out (b, chw), score (chw, k)
        '''
        
        b, c, h, w = x.size()        
        assert c == self.c    
        k, c = self.k, self.c
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (n, c)
        
        m1 = self.feat_units.weight.data # (k, c)
        m2 = self.label_units.weight.data # (k, 4)

        if update_flag and y is not None:
            y = y.permute(0, 2, 3, 1)
            y = y.reshape(-1, 4) # (n, 4)
            xn = torch.cat([F.normalize(x, dim=1), y], dim=1) # (n, c+4)
            mn = torch.cat([F.normalize(m1, dim=1), m2], dim=1) # (k, c+4)
            score = torch.matmul(xn, mn.t()) # (n, k)
            m1, m2 = self.update(x, y, score)
        else:
            xn = F.normalize(x, dim=1)
            mn = F.normalize(m1, dim=1)
            score = torch.matmul(xn, mn.t()) # (n, k)
        
        ### retrieval
        if update_flag == 'topk':
            weight = F.softmax(score, dim=1) 
            with torch.no_grad():
                batch_ind = torch.arange(h*w, device=x.device)[:,None]
                embed_ind = torch.sort(score, dim=1, descending=True)[1] # (n, k)
                embed_top = embed_ind[:, :int(0.1*self.k)] # (n, 0.1k)
            sel_wt = weight[batch_ind, embed_top] # (n, 0.1k)
            out_x = torch.einsum('nk,nkc->nc',sel_wt, m1[embed_top,:]) # (n, c)
            out_x = out_x.view(b, h, w, c).permute(0, 3, 1, 2)
            x = x.view(b, h, w, c).permute(0, 3, 1, 2)
            out_y = torch.einsum('nk,nkc->nc',sel_wt, m2[embed_top,:])# (n, 4)
            out_y = out_y.view(b, h, w, 4).permute(0, 3, 1, 2)
            out = torch.cat([x,out_x,out_y], dim=1) # (b, 2c+4, h, w)

        elif update_flag == 'all':
            weight = F.softmax(score, dim=1)
            out_x = torch.matmul(weight, m1) # (n, c)
            out_x = out_x.view(b, h, w, c).permute(0, 3, 1, 2)
            x = x.view(b, h, w, c).permute(0, 3, 1, 2)
            out_y = torch.matmul(weight, m2) # (n, 4)
            out_y = out_y.view(b, h, w, 4).permute(0, 3, 1, 2)
            out = torch.cat([x,out_x,out_y], dim=1) # (b, 2c+4, h, w)

        return out, score