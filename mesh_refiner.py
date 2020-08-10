import pprint

import torch
from torch.nn import functional as F
import torch.optim as optim
import pytorch3d.structures
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturedSoftPhongShader,
    SoftPhongShader
)
from tqdm import tqdm
import pandas as pd

from utils import utils, network_utils
from deformation.deformation_net import DeformationNetwork


class MeshRefiner():

    def __init__(self, cfg_yaml_path, device):
        self.cfg = utils.load_config(cfg_yaml_path)
        self.device = device

        self.num_iterations = self.cfg["training"]["num_iterations"]
        self.l2_lam = self.cfg["training"]["l2_lam"]


    # given a mesh, mask, and pose, solves an optimization problem which encourages
    # silhouette consistency on the mask at the given pose.
    # TODO: fix mesh (currently, needs to already be in device)
    # TODO: should the input be image or mask?
    def refine_mesh(self, mesh, image, mask, pred_dist, pred_elev, pred_azim):
        '''
        Args:
        pred_dist (int)
        pred_elev (int)
        pred_azim (int)
        image (np int array, 224 x 224 x 3, rgb, 0-255)
        mask (np bool array, 224 x 224)
            
        '''

        # prep inputs 
        pose_in = torch.unsqueeze(torch.tensor([pred_dist, pred_elev, pred_azim]),0).to(self.device)
        # prep for image input: normalize, set as floattensor, permute so channel is first, and turn into minibatch of size one
        image_in = torch.unsqueeze(torch.tensor(image/255, dtype=torch.float).permute(2,0,1),0).to(self.device)
        # preps for points input: set minibatch of size one
        verts_in = torch.unsqueeze(mesh.verts_packed(),0).to(self.device)
        mask_gt = torch.tensor(mask, dtype=torch.float).to(self.device)
        R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim) 
        num_vertices = mesh.verts_packed().shape[0]
        zero_deformation_tensor = torch.zeros((num_vertices, 3)).to(self.device)

        # prep network
        deform_net = DeformationNetwork(self.cfg, num_vertices, self.device)
        deform_net.to(self.device)

        optimizer = optim.Adam(deform_net.parameters(), lr=1e-4)

        # optimizing  
        loss_info = pd.DataFrame()
        for i in tqdm(range(self.num_iterations)):
            deform_net.train()
            optimizer.zero_grad()


            delta_v = deform_net(pose_in, image_in, verts_in)
            delta_v = delta_v.reshape((-1,3))
            deformed_mesh = mesh.offset_verts(delta_v)
    
            rendered_deformed_mesh = utils.render_mesh(deformed_mesh, R, T, self.device, img_size=224, silhouette=True)
            sil_loss = F.binary_cross_entropy(rendered_deformed_mesh[0, :,:, 3], mask_gt)
            l2_loss = F.mse_loss(delta_v, zero_deformation_tensor)
            loss = sil_loss + l2_loss * self.l2_lam

            loss.backward()
            optimizer.step()


            iter_loss_info = {"iter":i, "sil_loss": sil_loss.item(), "l2_loss": l2_loss.item(), "total_loss": loss.item()}
            loss_info = loss_info.append(iter_loss_info, ignore_index = True)

        return deformed_mesh, loss_info
            