import os
from tqdm import trange
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch import distributions as dist
import numpy as np
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, cfg, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False, uda_type=None, num_epochs=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.uda_type = uda_type
        self.cfg = cfg
        
        self.curr_epoch = 0
        self.num_epochs = num_epochs

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary (minibatch)
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict


    def uda_evaluate(self, uda_loader):
        device = self.device
        uda_accuracies = []

        for data in tqdm(uda_loader):
            with torch.no_grad():
                domain_pred_source, domain_labels_source, domain_pred_target, domain_labels_target = self.compute_uda(data)
                source_correct_vec = (torch.sigmoid(domain_pred_source) > 0.5) == domain_labels_source.byte()
                target_correct_vec = (torch.sigmoid(domain_pred_target) > 0.5) == domain_labels_target.byte()
                uda_accuracies.append(source_correct_vec.cpu().numpy())
                uda_accuracies.append(target_correct_vec.cpu().numpy())

        uda_avg_acc = np.mean(np.concatenate(uda_accuracies, axis = 0))
        return uda_avg_acc


    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))


    # c is the encoded inputs for the source domain; it can be supplied, if it was already precomputed before
    def compute_uda(self, data, c = None):
        device = self.device
        p = data.get('points').to(device)

        if c is None:
            # encoded image (source domain)
            inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
            c = self.model.encode_inputs(inputs)

        # encoded image (target domain)
        inputs_target = data.get('inputs_target_domain', torch.empty(p.size(0), 0)).to(device)
        c_target = self.model.encode_inputs(inputs_target)

        # source domain has label 0, target domain has label 1
        domain_pred_source = self.model.dann_discriminator_pred(c)
        domain_pred_target = self.model.dann_discriminator_pred(c_target)
        batch_size = c.shape[0]
        domain_labels_source = torch.zeros((batch_size, 1)).to(device)
        domain_labels_target = torch.ones((batch_size, 1)).to(device)

        return domain_pred_source, domain_labels_source, domain_pred_target, domain_labels_target




    def compute_loss(self, data):
        ''' Computes the overall loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}
        # encoded image (source domain)
        c = self.model.encode_inputs(inputs)

        # predicting & sampling from normal distribution for generative
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence loss. Note that this is always 0 if in the config.yaml encoder_latent: null and z_dim:0 
        # it's zero because q_z and p0_z will both be standard multivariate normal distributions
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # Domain Adaptation Loss
        if self.uda_type == "dann":

            if self.num_epochs is None:
                raise ValueError("For DANN domain adaptation, must specify num_epochs")

            # choosing weight based on config.
            if self.cfg["training_uda_dann"]["uda_train_style"] == "exp":
                lam = (2 / (1 + np.exp(-10 * (self.curr_epoch / self.num_epochs))) ) - 1
            elif self.cfg["training_uda_dann"]["uda_train_style"] == "binary":
                if self.curr_epoch > self.cfg["training_uda_dann"]["uda_epoch_begin"]:
                    lam = 1
                else:
                    lam = 0
            else:
                raise ValueError("DANN DA style needs to be either exp or binary.")

            domain_pred_source, domain_labels_source, domain_pred_target, domain_labels_target = self.compute_uda(data, c)
            uda_loss = F.binary_cross_entropy_with_logits(domain_pred_source, domain_labels_source) * lam + \
                F.binary_cross_entropy_with_logits(domain_pred_target, domain_labels_target) * lam
            #print("{} {} uda_loss: {}, lam: {}".format(self.curr_epoch, self.cfg["training_uda_dann"]["uda_epoch_begin"], uda_loss, lam))
            loss = loss + uda_loss



        # General points
        logits = self.model.decode(p, z, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss
