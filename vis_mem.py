import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
from sklearn.cluster import KMeans

# model = torch.load('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/MOSS/ddn_ssir.pth', map_location='cpu')
# mem=model['memory.units.weight']
# import pdb; pdb.set_trace()


model = torch.load('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/vitmae_cnp_contrast/ep12_jade2_contrast_softmlp_wo_mask.pth', map_location='cpu')
model = model['state_dict']
mem = model['bbox_head.mem.feat_units.weight'][:,-4:]

# col_embed = model['bbox_head.mem.pos.col_embed.weight']
# row_embed = model['bbox_head.mem.pos.row_embed.weight']


# x_emb = col_embed #120, 256
# y_emb = row_embed #120, 256
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# im = ax.imshow(y_emb)

# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(x_emb.size(1)), 8)
# ax.set_yticks(np.arange(x_emb.size(0)), 8)
# ax.set_title("Visualization of Position Embedding")
# fig.tight_layout()
# plt.savefig('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/vitmae_uncert/pos_emb_y.jpg')


# mem = F.normalize(mem, dim=1)
from sklearn.manifold import TSNE
tsne=TSNE(n_components=2, random_state=0)
mem_2d = tsne.fit_transform(mem.numpy())
import matplotlib.pyplot as plt
import matplotlib.cm as cm
color = cm.rainbow(np.linspace(0, 1, mem_2d.shape[0]))
plt.scatter(mem_2d[:, 0], mem_2d[:, 1], c=color)
plt.savefig('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/vitmae_cnp_contrast/cnp_wo_mask_y.jpg')
# import pdb; pdb.set_trace()
# img = np.zeros((360, 640, 3), dtype=np.uint8)
# os.makedirs('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/vis_mem', exist_ok=True)
# x_scale = mem[:, 0].max()
# y_scale = mem[:, 1].max()
# l_scale = mem[:, 2].max()
# t_scale = mem[:, 3].max()
# r_scale = mem[:, 4].max()
# b_scale = mem[:, 5].max()
# for id, m in enumerate(mem):
#     x, y, l, t, r, b = m
#     x1 = ((x/x_scale-l/l_scale)*320).clamp(0, 640).item()
#     y1 = ((y/y_scale-t/t_scale)*180).clamp(0, 360).item()
#     x2 = ((x/x_scale+r/r_scale)*320).clamp(0, 640).item()
#     y2 = ((y/y_scale+b/b_scale)*180).clamp(0, 360).item()
#     # img = np.zeros((360, 640, 3), dtype=np.uint8)
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
# cv2.imwrite(os.path.join('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/vitmae_uncert', 'uncert_proj'+'.jpg'), img)




# mem = np.load('/media/data/coky/OVIS/CMaskTrack-RCNN/corpus_box.npy')
# k_cluster = KMeans(n_clusters=128, random_state=0, tol=1e-2, max_iter=30).fit(mem)

# img = np.zeros((360, 640, 3), dtype=np.uint8)
# k_centers = k_cluster.cluster_centers_
# os.makedirs('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/vitmae_FPS/mem', exist_ok=True)
# for id, box in enumerate(k_centers):
#     x1, y1, x2, y2 = box
#     x1 = np.clip(x1*640, 0, 640)
#     y1 = np.clip(y1*360, 0, 360)
#     x2 = np.clip(x2*640, 0, 640)
#     y2 = np.clip(y2*360, 0, 360)
#     # img = np.zeros((360, 640, 3), dtype=np.uint8)
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
# cv2.imwrite(os.path.join('/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/vitmae_FPS/mem', 'corpus'+'.jpg'), img)