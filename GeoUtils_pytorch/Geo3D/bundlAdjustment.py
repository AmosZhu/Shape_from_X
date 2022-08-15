"""
Author: dizhong zhu
Date: 12/08/2022

Bundle adjustment using pytorch optimisation under different constraint
"""
import torch
from GeoUtils_pytorch.Cameras.PerspectiveCamera import (
    BaseCamera,
    backProjection
)
from GeoUtils_pytorch.Geo3D.Rotation import (
    R_cleanUp
)
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix
)
from tqdm import tqdm
import copy


def RtsPC_f_BA_K(cameras: BaseCamera,
                 px: torch.Tensor, pc_init: torch.Tensor,
                 px_mask: torch.Tensor = None, pc_mask: torch.Tensor = None,
                 message=''):
    '''
    By fixing intrisic parameters, we are aiming to minimise the gemoetric error between:
        back projected 3D points and pixel coordinates in each view.

    @ Optimise:
        - rotation
        - translation
        - scale
        - 3D point cloud

    You might expect the px in each view have different length, but for covinient we take a tensor,
        please fill what ever you want in missing part and sepcify with a mask.

    :param cameras: [nView] cameras
    :param px:      2D pixel coordinate:            size=[nView, N,2]
    :param pc_init: 3D initial point cloud:         size=[N,3]
    :param px_mask: visibility mask for px          size=[nView, N]
    :param pc_mask: mask for point cloud            size=[N]
    :return: new cameras with optimised rotation, translation and scale.  3D point cloud
    '''
    nView = len(cameras)
    N = pc_init.shape[0]

    device = px.device

    px_mask = torch.ones(size=(nView, N), dtype=torch.bool).to(device) if px_mask is None else px_mask
    pc_mask = torch.ones(size=(N,), dtype=torch.bool).to(device) if pc_mask is None else pc_mask

    # if one row of the px_mask sum 0, means no view see this feature, we get rid of it.
    mask = px_mask.sum(0) > 0
    mask = mask & pc_mask

    # truncate the pixel and point cloud if the 3D points are not going to optimise anyway
    px_mask = px_mask[:, mask == True]
    px = px[:, mask == True]

    pc = pc_init[mask]
    pc.requires_grad = True

    pc_final = pc_init
    cam_final = copy.deepcopy(cameras)

    cam_final.upgrade_only(unfreeze_list=['R', 't', 's'])

    # set up the optimiser to optimise quternion and translation vector and point cloud

    criterion = torch.nn.MSELoss()
    lr = 5e-5
    bConverge = False
    p_loss = 1e8
    max_iteration = 10000
    verbose = True
    optimiser = torch.optim.Adam(params=[pc] + list(cam_final.parameters()), lr=lr)
    n_iter = tqdm(range(max_iteration), disable=not verbose)

    for i, it in enumerate(n_iter):
        if bConverge: break

        optimiser.zero_grad()

        bp_imgpt = cam_final.point_to_image(pc[None])

        loss = criterion(px[px_mask], bp_imgpt[px_mask])
        # loss = ((px[px_mask] - bp_imgpt[px_mask]) ** 2).sum(-1).mean()

        if (loss.data - p_loss).abs() < 1e-16:
            bConverge = True

        n_iter.set_description(f'{message}[Camera & PointCloud Bundle Adjustment {i}/{max_iteration}] loss: {loss.data:2f}')

        loss.backward()
        optimiser.step()

    pc_final[mask] = pc.detach()

    # keep the final point cloud output samesize

    return cam_final, pc_final


def Rts_f_BA_K(cameras: BaseCamera,
               px: torch.Tensor, pc_init: torch.Tensor,
               px_mask: torch.Tensor = None, pc_mask: torch.Tensor = None,
               message=''):
    '''
    By given intrisic parameters, we are aiming to minimise the gemoetric error between:
        back projected 3D points and pixel coordinates in each view.

    @ Optimise:
        - rotation
        - translation
        - scale

    You might expect the px in each view have different length, but for covinient we take a tensor,
        please fill what ever you want in missing part and sepcify with a mask.

    :param cameras: [nView] cameras
    :param px:      2D pixel coordinate:            size=[nView, N,2]
    :param pc_init: 3D initial point cloud:         size=[N,3]
    :param px_mask: visibility mask for px          size=[nView, N]
    :param pc_mask: mask for point cloud            size=[N]
    :return: new cameras with optimised rotation, translation and scale
    '''
    nView = len(cameras)
    N = pc_init.shape[0]

    device = px.device

    px_mask = torch.ones(size=(nView, N), dtype=torch.bool).to(device) if px_mask is None else px_mask
    pc_mask = torch.ones(size=(N,), dtype=torch.bool).to(device) if pc_mask is None else pc_mask

    # truncate the pixel and point cloud if the 3D points are not going to optimise anyway
    px_mask = px_mask[:, pc_mask == True]
    px = px[:, pc_mask == True]

    pc = pc_init[pc_mask]

    cam_final = copy.deepcopy(cameras)
    cam_final.upgrade_only(unfreeze_list=['R', 't', 's'])

    # set up the optimiser to optimise quternion and translation vector and point cloud

    lr = 1e-4
    bConverge = False
    p_loss = 1e8
    max_iteration = 10000
    verbose = True
    optimiser = torch.optim.Adam(params=cam_final.parameters(), lr=lr)
    n_iter = tqdm(range(max_iteration), disable=not verbose)
    criterion = torch.nn.MSELoss()

    for i, it in enumerate(n_iter):
        if bConverge: break

        optimiser.zero_grad()

        bp_imgpt = cam_final.point_to_image(pc[None])

        # loss = ((px[px_mask] - bp_imgpt[px_mask]) ** 2).sum(-1).mean()

        loss = criterion(px[px_mask], bp_imgpt[px_mask])

        if (loss.data - p_loss).abs() < 1e-16:
            bConverge = True

        p_loss = loss.data

        n_iter.set_description(f'{message}[Camera Bundle Adjustment {i}/{max_iteration}] loss: {loss.data:2f}')

        loss.backward()
        optimiser.step()

    return cam_final


#############################################################################################################
#
#   These are the functions when I prototype, it might be good for reference,
#   they are similar as above function where I use my own camera class instead tensor directly.
#   It's still useful, because I don't want to take too many arguments, so I have a camera clss above
#
#   !!!!!!! Be aware, I didn't use scale here, so might only work if you are using pair of cameras
#
#############################################################################################################
def RtPC_f_BA_K(K: torch.Tensor, R_init: torch.Tensor, t_init: torch.Tensor,
                px: torch.Tensor, pc_init: torch.Tensor,
                px_mask: torch.Tensor = None, pc_mask: torch.Tensor = None):
    '''
    By given intrisic parameters, we are aiming to minimise the gemoetric error between:
        back projected 3D points and pixel coordinates in each view.

    You might expect the px in each view have different length, but for covinient we take a tensor,
        please fill what ever you want in missing part and sepcify with a mask.

    :param K:       Intrinsic parameters:           size=[nView,3,3]
    :param R_init:  Initial rotation matrices:      size=[nView,3,3]
    :param t_init:  Initial translation vectors:    size=[nView, 3]
    :param px:      2D pixel coordinate:            size=[nView, N,2]
    :param pc_init: 3D initial point cloud:         size=[N,3]
    :param px_mask: visibility mask for px          size=[nView, N]
    :param pc_mask: mask for point cloud            size=[N]
    :return: R,t and PC: rotation , translation and point cloud
    '''
    assert K.shape[0] == R_init.shape[0] and K.shape[0] == t_init.shape[0]
    nView = K.shape[0]
    N = pc_init.shape[0]

    device = K.device

    px_mask = torch.ones(size=(nView, N), dtype=torch.bool).to(device) if px_mask is None else px_mask
    pc_mask = torch.ones(size=(N,), dtype=torch.bool).to(device) if pc_mask is None else pc_mask

    # if one row of the px_mask sum 0, means no view see this feature, we get rid of it.
    mask = px_mask.sum(0) > 0
    mask = mask & pc_mask

    # truncate the pixel and point cloud if the 3D points are not going to optimise anyway
    px_mask = px_mask[:, mask == True]
    px = px[:, mask == True]

    q = matrix_to_quaternion(R_init)
    t = t_init
    pc = pc_init[mask]
    pc_final = pc_init

    pc.requires_grad = True
    q.requires_grad = True
    t.requires_grad = True

    # set up the optimiser to optimise quternion and translation vector and point cloud

    lr = 1e-5
    max_iteration = 5000
    verbose = True
    optimiser = torch.optim.Adam(params=[q, t, pc], lr=lr)
    n_iter = tqdm(range(max_iteration), disable=not verbose)

    for i, it in enumerate(n_iter):
        optimiser.zero_grad()

        R = quaternion_to_matrix(q)
        P = K @ torch.cat([R, t[..., None]], dim=-1)

        bp_imgpt = backProjection(P, pc)

        loss = ((px[px_mask] - bp_imgpt[px_mask]) ** 2).sum().sqrt()

        n_iter.set_description(f'[Bundle Adjustment {i}/{max_iteration}] loss: {loss.data:2f}')

        loss.backward()
        optimiser.step()

    R_final = quaternion_to_matrix(q).detach()
    t_final = t.detach()
    pc_final[mask] = pc.detach()

    # keep the final point cloud output samesize

    return R_final, t_final, pc_final


def Rt_f_BA_K(K: torch.Tensor, R_init: torch.Tensor, t_init: torch.Tensor,
              px: torch.Tensor, pc_init: torch.Tensor,
              px_mask: torch.Tensor = None, pc_mask: torch.Tensor = None):
    '''
    By given intrisic parameters, we are aiming to minimise the gemoetric error between:
        back projected 3D points and pixel coordinates in each view.

    You might expect the px in each view have different length, but for covinient we take a tensor,
        please fill what ever you want in missing part and sepcify with a mask.

    :param K:       Intrinsic parameters:           size=[nView,3,3]
    :param R_init:  Initial rotation matrices:      size=[nView,3,3]
    :param t_init:  Initial translation vectors:    size=[nView, 3]
    :param px:      2D pixel coordinate:            size=[nView, N,2]
    :param pc_init: 3D initial point cloud:         size=[N,3]
    :param px_mask: visibility mask for px          size=[nView, N]
    :param pc_mask: mask for point cloud            size=[N]
    :return: R,t: rotation and translation
    '''
    assert K.shape[0] == R_init.shape[0] and K.shape[0] == t_init.shape[0]
    nView = K.shape[0]
    N = pc_init.shape[0]

    device = K.device

    px_mask = torch.ones(size=(nView, N), dtype=torch.bool).to(device) if px_mask is None else px_mask
    pc_mask = torch.ones(size=(N,), dtype=torch.bool).to(device) if pc_mask is None else pc_mask

    # truncate the pixel and point cloud if the 3D points are not going to optimise anyway
    px_mask = px_mask[:, pc_mask == True]
    px = px[:, pc_mask == True]

    # q = matrix_to_quaternion(R_init)
    q = R_init
    t = t_init
    pc = pc_init[pc_mask]

    q.requires_grad = True
    t.requires_grad = True

    # set up the optimiser to optimise quternion and translation vector and point cloud

    lr = 5e-4
    bConverge = False
    p_loss = 1e8
    max_iteration = 10000
    verbose = True
    optimiser = torch.optim.Adam(params=[q, t], lr=lr)
    n_iter = tqdm(range(max_iteration), disable=not verbose)

    for i, it in enumerate(n_iter):
        if bConverge: break

        optimiser.zero_grad()

        # R = quaternion_to_matrix(q)

        R = q
        P = K @ torch.cat([R, t[..., None]], dim=-1)

        bp_imgpt = backProjection(P, pc)

        loss_pix = ((px[px_mask] - bp_imgpt[px_mask]) ** 2).sum().sqrt()
        loss_R = (R @ R.transpose(1, 2) - torch.eye(3)[None, ...]).abs().mean() * 10000

        loss = loss_pix + loss_R
        # if (loss.data - p_loss).abs() < 1e-16:
        #     bConverge = True

        p_loss = loss.data

        n_iter.set_description(f'[Bundle Adjustment {i}/{max_iteration}] loss: {loss.data:2f}, loss_pix: {loss_pix.data:2f},loss_R: {loss_R.data:2f}')

        loss.backward()
        optimiser.step()

    # R_final = quaternion_to_matrix(q).detach()
    R_final = R_cleanUp(q).detach()
    t_final = t.detach()

    return R_final, t_final
