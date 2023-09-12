import torch
import torch.nn as nn
import torch.nn.functional as F
from .MLP import MLP
import numpy as np


# ## cnp mean with normed pos
# class MemoryN2N(nn.Module):        
#     def __init__(self, 
#                  hdim, 
#                  kdim,
#                  moving_average_rate=0.999,
#                  update_mode='top1'):
#         super().__init__()
        
#         self.c = hdim
#         self.k = kdim
#         self.topk = int(0.1*self.k)
#         self.moving_average_rate = moving_average_rate
#         self.update_mode = update_mode
#         self.feat_units = nn.Embedding(kdim, hdim+6) # (k, c+4)
#         self.mlp = MLP(hdim+6, hdim, hdim, drop=0.1)

#     def update(self, xy, score):
#         '''
#             xy: (n, c+4)
#             score: (n, k)
#         '''
        
#         m = self.feat_units.weight.data # (k, c+4)
        
#         xy = xy.detach()

#         if self.update_mode == 'top1':
#             embed_ind = torch.max(score, dim=1)[1] # (n, )
#             embed_onehot = F.one_hot(embed_ind, self.k).type(xy.dtype) # (n, k)        
#         elif self.update_mode == 'topk':
#             topk_score, topk_ind = torch.topk(score, self.topk, dim=1)
#             score_soft = torch.softmax(score, dim=1) # (n, k) 
#             embed_onehot = score.new_zeros(score.size()) # (n, k)   
#             embed_onehot[topk_ind] = score_soft[topk_ind] # (n, k)
#         elif self.update_mode == 'all':
#             embed_onehot = torch.softmax(score, dim=1) # (n, k)   

#         embed_onehot_sum = embed_onehot.sum(0)

#         embed_sum = xy.transpose(0, 1) @ embed_onehot # (c+4, k)
#         embed_mean = embed_sum / (embed_onehot_sum + 1e-6)

#         new_feat_units = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)

#         if self.training:
#             self.feat_units.weight.data = new_feat_units

#         return new_feat_units
                
#     def forward(self, x, pos, y=None, mask=None, update_flag=True):
#         '''
#           x: feats of shape(b, c, h, w)
#           pos: normalised position of shpae (h,w,2)
#           y: gt of shape (b, 4, h, w) in xyxy format in range [0, 1]
#           mask: binary mask of shape (hw) to filter out out-of-box pixels
#           embed: (k, c)
#           output: out (b, chw), score (chw, k)
#         '''
#         with torch.no_grad():
#             b, c, h, w = x.size()        
#             assert c == self.c    
#             k, c = self.k, self.c
            
#             x = x.permute(0, 2, 3, 1)
#             x = x.reshape(-1, c) # (N, c)
#             pos = pos.reshape(-1, 2) # (hw, 2)
#             pos = pos.repeat(b, 1) # (N, 2)
#             x = torch.cat([x, pos], dim=1) # (N, c+2)
#             xn = F.normalize(x, dim=1) # (N, c+2)
#             m = self.feat_units.weight.data[:, :-4] # (k, c+2)
#             mn = F.normalize(m, dim=1) # (k, c+2)
#             score = torch.matmul(xn, mn.t())# (N, k)

#             if update_flag and y is not None and mask is not None:
#                 y = y.permute(0, 2, 3, 1)
#                 y = y.reshape(-1, 4) # (N, 4)
#                 xy = torch.cat([x, y], dim=1) # (N, c+6)

#                 ### only update points within a box
#                 new_units = self.update(xy[mask], score[mask]) # (k, c+6)
                

#         # deterministic path
#         out_r = self.mlp(self.feat_units.weight.data) # (k, c)
#         out_r = out_r.mean(0) #c
#         out_r = out_r.view(1, -1, 1, 1).repeat(b, 1, h, w) #b,c,h,w
#         return out_r



# cnp mean
class MemoryN2N(nn.Module):        
    def __init__(self, hdim, kdim,
                 moving_average_rate=0.999,
                 update_mode='top1'):
        super().__init__()
        
        self.c = hdim
        self.k = kdim
        self.topk = int(0.1*self.k)
        self.moving_average_rate = moving_average_rate
        
        self.feat_units = nn.Embedding(kdim, hdim+4) # (k, c+4)
        self.mlp = MLP(hdim+4, hdim, hdim, drop=0.1)
        self.update_mode = update_mode

    def update(self, xy, score):
        '''
            xy: (n, c+4)
            score: (n, k)
        '''
        
        m = self.feat_units.weight.data # (k, c+4)
        
        xy = xy.detach()

        if self.update_mode == 'top1':
            embed_ind = torch.max(score, dim=1)[1] # (n, )
            embed_onehot = F.one_hot(embed_ind, self.k).type(xy.dtype) # (n, k)        
        elif self.update_mode == 'topk':
            topk_score, topk_ind = torch.topk(score, self.topk, dim=1)
            score_soft = torch.softmax(score, dim=1) # (n, k) 
            embed_onehot = score.new_zeros(score.size()) # (n, k)   
            embed_onehot[topk_ind] = score_soft[topk_ind] # (n, k)
        elif self.update_mode == 'all':
            embed_onehot = torch.softmax(score, dim=1) # (n, k)   

        embed_onehot_sum = embed_onehot.sum(0)

        embed_sum = xy.transpose(0, 1) @ embed_onehot # (c+4, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)

        new_feat_units = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)

        if self.training:
            self.feat_units.weight.data = new_feat_units

        return new_feat_units
    
    def get_topk(self, score, topk):
        '''
            score: (k)
            topk: int
        '''
        topk_score, topk_ind = torch.topk(score, topk)
        top_mems = self.feat_units.weight.data[topk_ind, -4:] # (topk, 4)
        return top_mems
                
    def forward(self, x, y=None, mask=None, update_flag=True):
        '''
          x: feats of shape(b, c, h, w)
          y: gt of shape (b, 4, h, w) in xyxy format in range [0, 1]
          mask: binary mask of shape (hw) to filter out out-of-box pixels
          embed: (k, c)
          output: out (b, chw), score (chw, k)
        '''
        b, c, h, w = x.size()        
        assert c == self.c    
        k, c = self.k, self.c
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (N, c)
        xn = F.normalize(x, dim=1) # (N, c)
        m = self.feat_units.weight.data[:, :-4] # (k, c)
        mn = F.normalize(m, dim=1) # (k, c)
        score = torch.matmul(xn, mn.t())# (N, k)

        with torch.no_grad():
            m = self.feat_units.weight.data[:, :-4] # (k, c)

            if update_flag and y is not None and mask is not None:
                y = y.permute(0, 2, 3, 1)
                y = y.reshape(-1, 4) # (N, 4)
                xy = torch.cat([x, y], dim=1) # (N, c+4)

                ### only update points within a box
                new_units = self.update(xy[mask], score[mask]) # (k, c+4)
            elif isinstance(update_flag, tuple) or isinstance(update_flag, list):
                i, j = update_flag
                score = score.reshape(b, h, w, -1).squeeze(0)
                top_items = self.get_topk(score[i,j], 5) # (5, 4)
                return top_items
                

        # deterministic path
        out_r = self.mlp(self.feat_units.weight.data) # (k, c)
        out_r = out_r.mean(0) #c
        out_r = out_r.view(1, -1, 1, 1).repeat(b, 1, h, w) #b,c,h,w
        return out_r





# # ### softmax retrieval
# class MemoryN2N(nn.Module):        
#     def __init__(self, 
#                  hdim, 
#                  kdim,
#                  moving_average_rate=0.999,
#                  update_mode='top1'):
#         super().__init__()

#         self.c = hdim
#         self.k = kdim
#         self.topk = int(0.1*kdim)
#         self.moving_average_rate = moving_average_rate
        
#         self.feat_units = nn.Embedding(kdim, hdim+4) # (k, c+4)

#         self.mlp = MLP(hdim+4, hdim, hdim, drop=0.)
#         self.update_mode = update_mode

#     def update(self, xy, score):
#         '''
#             xy: (n, c+4)
#             score: (n, k)
#         '''
#         m = self.feat_units.weight.data # (k, c+4)
        
#         xy = xy.detach()

#         if self.update_mode == 'top1':
#             embed_ind = torch.max(score, dim=1)[1] # (n, )
#             embed_onehot = F.one_hot(embed_ind, self.k).type(xy.dtype) # (n, k)        
#         elif self.update_mode == 'topk':
#             topk_score, topk_ind = torch.topk(score, self.topk, dim=1)
#             score_soft = torch.softmax(score, dim=1) # (n, k) 
#             embed_onehot = score.new_zeros(score.size()) # (n, k)   
#             embed_onehot[topk_ind] = score_soft[topk_ind] # (n, k)
#         elif self.update_mode == 'all':
#             embed_onehot = torch.softmax(score, dim=1) # (n, k)   

#         embed_onehot_sum = embed_onehot.sum(0)

#         embed_sum = xy.transpose(0, 1) @ embed_onehot # (c+4, k)
#         embed_mean = embed_sum / (embed_onehot_sum + 1e-6)

#         new_feat_units = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)

#         if self.training:
#             self.feat_units.weight.data = new_feat_units

#         return new_feat_units
                
#     def forward(self, x, y=None, mask=None, update_flag=True):
#         '''
#           x: feats of shape(b, c, h, w)

#           y: gt of shape (b, 4, h, w) in xyxy format in range [0, 1]
#           mask: binary mask of shape (hw) to filter out out-of-box pixels
#           embed: (k, c)
#           output: out (b, chw), score (chw, k)
#         '''

#         b, c, h, w = x.size()        
#         x_flat = x.permute(0, 2, 3, 1)
#         x_flat = x_flat.reshape(-1, c) # (N, c)
#         xn = F.normalize(x_flat, dim=1) # (N, c)
#         m = self.feat_units.weight.data[:, :-4] # (k, c)
#         mn = F.normalize(m, dim=1) # (k, c)
#         score = torch.matmul(xn, mn.t())# (N, k)

#         m = self.feat_units.weight.data[:, :-4] # (k, c-4)

#         if update_flag and y is not None:
#             y = y.permute(0, 2, 3, 1)
#             y = y.reshape(-1, 4) # (N, 4)
#             xy = torch.cat([x_flat, y], dim=1) # (N, c+4)

#             ### only update points within a box
#             m = self.update(xy, score) # (k, c+4)
                
#         # deterministic path
#         score = torch.softmax(score, dim=1) # (N, k)
#         out_r = score @ self.feat_units.weight.data # (N, c+4)
#         out_r = self.mlp(out_r) # (N, c)
#         out_r = out_r.reshape(b, h, w, -1).permute(0, 3, 1, 2) # (b, c, h, w)
#         return out_r


# # ### cnp mean wo pos wo y
# class MemoryN2N(nn.Module):        
#     def __init__(self, 
#                  hdim, 
#                  kdim,
#                  moving_average_rate=0.999,
#                  update_mode='top1'):
#         super().__init__()

#         self.c = hdim
#         self.k = kdim
#         self.topk = int(0.1*kdim)
#         self.moving_average_rate = moving_average_rate
        
#         self.feat_units = nn.Embedding(kdim, hdim) # (k, c+4)
#         self.mlp = MLP(hdim, hdim, hdim, drop=0.1)
#         self.update_mode = update_mode

#     def update(self, x, score):
#         '''
#             x: (n, c)
#             score: (n, k)
#         '''
#         m = self.feat_units.weight.data # (k, c+4)
        
#         x = x.detach()

#         if self.update_mode == 'top1':
#             embed_ind = torch.max(score, dim=1)[1] # (n, )
#             embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype) # (n, k)        
#         elif self.update_mode == 'topk':
#             topk_score, topk_ind = torch.topk(score, self.topk, dim=1)
#             score_soft = torch.softmax(score, dim=1) # (n, k) 
#             embed_onehot = score.new_zeros(score.size()) # (n, k)   
#             embed_onehot[topk_ind] = score_soft[topk_ind] # (n, k)
#         elif self.update_mode == 'all':
#             embed_onehot = torch.softmax(score, dim=1) # (n, k)   

#         embed_onehot_sum = embed_onehot.sum(0)

#         embed_sum = x.transpose(0, 1) @ embed_onehot # (c+4, k)
#         embed_mean = embed_sum / (embed_onehot_sum + 1e-6)

#         new_feat_units = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)

#         if self.training:
#             self.feat_units.weight.data = new_feat_units

#         return new_feat_units
                
#     def forward(self, x, y=None, mask=None, update_flag=True):
#         '''
#           x: feats of shape(b, c, h, w)
#           y: gt of shape (b, 4, h, w) in xyxy format in range [0, 1]
#           mask: binary mask of shape (nhw) to filter out out-of-box pixels
#           embed: (k, c)
#           output: out (b, chw), score (chw, k)
#         '''

#         b, c, h, w = x.size()        
#         x_flat = x.permute(0, 2, 3, 1)
#         x_flat = x_flat.reshape(-1, c) # (N, c)
#         xn = F.normalize(x_flat, dim=1) # (N, c)
#         m = self.feat_units.weight.data # (k, c)
#         mn = F.normalize(m, dim=1) # (k, c)
#         score = torch.matmul(xn, mn.t())# (N, k)

#         m = self.feat_units.weight.data # (k, c-4)

#         if update_flag and mask is not None:
#             ### only update points within a box
#             m = self.update(x_flat[mask], score[mask]) # (k, c+4)
                
#         # deterministic path
#         out_r = self.mlp(self.feat_units.weight.data) # (k, c)
#         out_r = out_r.mean(0) #c
#         out_r = out_r.view(1, -1, 1, 1).repeat(b, 1, h, w) #b,c,h,w
#         return out_r