import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.nn import functional as F
import pytorch3d
from pytorch3d.renderer import look_at_view_transform
from torchvision import transforms
import pickle

from utils import utils
import sys
if 'occ_uda' not in sys.path:
    sys.path.insert(1, 'occ_uda')
from occ_uda.im2mesh import config
from im2mesh.checkpoints import CheckpointIO


class SemanticEmbeddingLoss():
    def __init__(self, cfg_yaml_path, weights_path, training_embeddings_path, device):
        self.training_embeddings = pickle.load(open(training_embeddings_path, "rb"))
        self.nbrs_struct = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.training_embeddings.detach().cpu())
        self.device = device

        # setting up occnet
        cfg = config.load_config(cfg_yaml_path, 'occ_uda/configs/default.yaml')
        self.model = config.get_model(cfg, device=device)
        checkpoint_io = CheckpointIO('/'.join(weights_path.split('/')[:-1]), model=self.model)
        checkpoint_io.load(weights_path.split('/')[-1], device)
        self.generator = config.get_generator(self.model, cfg, device=device)
        self.model.eval()
        # freeze weights
        for param in self.model.parameters():
            param.requires_grad = False


    # for debug/visualization purposes only
    def generate_latent_mesh(self, c, mesh_save_path):
        z = self.model.get_z_from_prior((1,), sample=False).to(self.device)
        mesh = self.generator.generate_from_latent(z, c)
        mesh.export(mesh_save_path)


    def compute_loss(self, mesh):
        azim = 151.57894897460938
        dist = 1.269230842590332
        elev =  9.473684310913086

        R, T = look_at_view_transform(dist, elev, azim) 
        # occnet takes in inputs of size [1, 3, 224, 224]
        render = utils.render_mesh(mesh, R, T, self.device, img_size=224)
        render = render[...,:3].permute(0,3,1,2)

        c = self.model.encode_inputs(render)

        distances, indices = self.nbrs_struct.kneighbors(c.detach().cpu())
        nearest_embedding = self.training_embeddings[indices[0]]
        self.generate_latent_mesh(nearest_embedding, "nearest_emb.obj")
        
        #self.generate_latent_mesh(c, "reconstructed.obj")

        loss = F.mse_loss(c, torch.unsqueeze(self.training_embeddings[0],0))

        return loss, render




