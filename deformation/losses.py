import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.nn import functional as F
import pytorch3d
from pytorch3d.renderer import look_at_view_transform
from torchvision import transforms

from utils import utils

# given a mesh and a plane of symmetry (represented by its unit normal)
# computes the symmetry loss of the mesh, defined as the average
# distance between points on one side and its nearest neighbor point on the other side
# args:
#   Sym_plane: list of 3 numbers
# https://math.stackexchange.com/questions/693414/reflection-across-the-plane
def vertex_symmetry_loss_fast(mesh, sym_plane, device):
    N = np.array([sym_plane])
    if np.linalg.norm(N) != 1:
        raise ValueError("sym_plane needs to be a unit normal")

    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float).to(device)

    mesh_verts = mesh.verts_packed()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mesh_verts.detach().cpu())

    sym_points = mesh_verts @ reflect_matrix
    distances, indices = nbrs.kneighbors(sym_points.detach().cpu())
    #avg_sym_loss = F.mse_loss(sym_points, torch.squeeze(mesh_verts[indices],1), reduction='sum')
    avg_sym_loss = F.l1_loss(sym_points, torch.squeeze(mesh_verts[indices],1), reduction='sum')

    return avg_sym_loss


# image based symmetry loss
# renders mesh at offsets about the plane of symmetry and computes a MSE loss in pixel space
def image_symmetry_loss(mesh, sym_plane, device):
    N = np.array([sym_plane])
    if np.linalg.norm(N) != 1:
        raise ValueError("sym_plane needs to be a unit normal")

    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float).to(device)

    dist = 1.269230842590332
    elev = 9.473684310913086
    #elev = 50.473684310913086
    azim = 151.57894897460938
    #azim = 151.57894897460938
    R, T = look_at_view_transform(dist, elev, azim)

    # render at camera 1
    R1 = utils.render_mesh(mesh, R, T, device, img_size=224, silhouette=True)[0, :,:, 3]
    R1_flipped = torch.flip(R1, [1])
    # render at camera 2: camera 1 reflected across plane of symmetry
    camera_position = pytorch3d.renderer.cameras.camera_position_from_spherical_angles(dist, elev, azim).to(device)
    R_sym = pytorch3d.renderer.cameras.look_at_rotation(camera_position@reflect_matrix)
    R2 = utils.render_mesh(mesh, R_sym, T, device, img_size=224, silhouette=True)[0, :,:, 3]
    
    sym_loss = F.mse_loss(R1_flipped, R2)
    #sym_loss = F.binary_cross_entropy(R1_flipped, R2)
    
    return sym_loss
    #return sym_loss, R1, R1_flipped, R2



