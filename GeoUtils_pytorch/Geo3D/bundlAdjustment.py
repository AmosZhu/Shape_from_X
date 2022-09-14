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
from pytorch3d.renderer.cameras import PerspectiveCameras


def RtsPC_f_BA_K(cameras: BaseCamera,
                 px: torch.tensor, pc_init: torch.tensor,
                 px_mask: torch.tensor = None, pc_mask: torch.tensor = None,
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
    cam_final = cameras.detach()
    cam_final.s.requires_grad = True
    cam_final.t.requires_grad = True
    q = matrix_to_quaternion(cam_final.R.detach())
    q.requires_grad = True

    # set up the optimiser to optimise quternion and translation vector and point cloud

    criterion = torch.nn.MSELoss()
    lr = 5e-5
    bConverge = False
    p_loss = 1e8
    max_iteration = 10000
    verbose = True
    optimiser = torch.optim.Adam(params=[pc, q, cam_final.t, cam_final.s], lr=lr)
    n_iter = tqdm(range(max_iteration), disable=not verbose)

    for i, it in enumerate(n_iter):
        if bConverge: break

        optimiser.zero_grad()

        cam_final.R = quaternion_to_matrix(q)
        bp_imgpt = cam_final.point_to_image(pc[None])

        loss = criterion(px[px_mask], bp_imgpt[px_mask])

        if (loss.data - p_loss).abs() < 1e-16:
            bConverge = True

        n_iter.set_description(f'{message}[Camera & PointCloud Bundle Adjustment {i}/{max_iteration}] loss: {loss.data:2f}')

        loss.backward()
        optimiser.step()

    pc_final[mask] = pc.detach()

    # keep the final point cloud output samesize

    return cam_final, pc_final


def Rts_f_BA_K(cameras: BaseCamera,
               px: torch.tensor, pc: torch.tensor,
               px_mask: torch.tensor = None, pc_mask: torch.tensor = None,
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
    :param pc:      3D point cloud:                 size=[N,3]
    :param px_mask: visibility mask for px          size=[nView, N]
    :param pc_mask: mask for point cloud            size=[N]
    :return: new cameras with optimised rotation, translation and scale
    '''
    nView = len(cameras)
    N = pc.shape[0]

    device = px.device

    px_mask = torch.ones(size=(nView, N), dtype=torch.bool).to(device) if px_mask is None else px_mask
    pc_mask = torch.ones(size=(N,), dtype=torch.bool).to(device) if pc_mask is None else pc_mask

    # truncate the pixel and point cloud if the 3D points are not going to optimise anyway
    px_mask = px_mask[:, pc_mask == True]
    px = px[:, pc_mask == True]

    pc = pc[pc_mask]

    cam_final = cameras.detach()
    cam_final.s.requires_grad = True
    cam_final.t.requires_grad = True
    q = matrix_to_quaternion(cam_final.R.detach())
    q.requires_grad = True

    # set up the optimiser to optimise quternion and translation vector and point cloud

    lr = 1e-4
    bConverge = False
    p_loss = 1e8
    max_iteration = 10000
    verbose = True
    optimiser = torch.optim.Adam(params=[q, cam_final.t, cam_final.s], lr=lr)
    n_iter = tqdm(range(max_iteration), disable=not verbose)
    criterion = torch.nn.MSELoss()

    for i, it in enumerate(n_iter):
        if bConverge: break

        optimiser.zero_grad()

        cam_final.R = quaternion_to_matrix(q)

        bp_imgpt = cam_final.point_to_image(pc[None])
        loss = criterion(px[px_mask], bp_imgpt[px_mask])

        if (loss.data - p_loss).abs() < 1e-16:
            bConverge = True

        p_loss = loss.data

        n_iter.set_description(f'{message}[Camera Bundle Adjustment {i}/{max_iteration}] loss: {loss.data:2f}')

        loss.backward()
        optimiser.step()

    return cam_final


def fRt_f_BA_K(cameras: BaseCamera,
               px: torch.tensor, pc: torch.tensor,
               px_mask: torch.tensor = None, pc_mask: torch.tensor = None,
               message=''):
    '''
    we are aiming to minimise the gemoetric error between:
        back projected 3D points and pixel coordinates in each view.

    @ Optimise:
        - intrinsic parameters (focal length only)
        - rotation
        - translation
        - scale

    You might expect the px in each view have different length, but for covinient we take a tensor,
        please fill what ever you want in missing part and sepcify with a mask.

    :param cameras: [nView] cameras
    :param px:      2D pixel coordinate:            size=[nView, N,2]
    :param pc:      3D point cloud:                 size=[N,3]
    :param px_mask: visibility mask for px          size=[nView, N]
    :param pc_mask: mask for point cloud            size=[N]
    :return: new cameras with optimised rotation, translation and scale
    '''
    nView = len(cameras)
    N = pc.shape[0]

    device = px.device

    px_mask = torch.ones(size=(nView, N), dtype=torch.bool).to(device) if px_mask is None else px_mask
    pc_mask = torch.ones(size=(N,), dtype=torch.bool).to(device) if pc_mask is None else pc_mask

    # truncate the pixel and point cloud if the 3D points are not going to optimise anyway
    px_mask = px_mask[:, pc_mask == True]
    px = px[:, pc_mask == True]

    pc = pc[pc_mask]

    cam_final = cameras.detach()
    K_init = cam_final.K.detach()[0]
    f = K_init[0, 0]
    f.requires_grad = True
    cam_final.t.requires_grad = True
    q = matrix_to_quaternion(cam_final.R.detach())
    q.requires_grad = True

    # set up the optimiser to optimise quternion, translation vector and focal length

    lr = 2e-1
    bConverge = False
    p_loss = 1e8
    max_iteration = 10000
    verbose = True
    optimiser = torch.optim.Adam(params=[f, cam_final.t], lr=lr)
    n_iter = tqdm(range(max_iteration), disable=not verbose)
    criterion = torch.nn.MSELoss()

    for i, it in enumerate(n_iter):
        if bConverge: break

        optimiser.zero_grad()
        cam_final.R = quaternion_to_matrix(q)

        view_pc = cam_final.point_to_view(pc[None])
        # project the points to the image space use focal length formula
        bp_imgpt_x = f / view_pc[..., 2] * view_pc[..., 0] + K_init[0, 2]
        bp_imgpt_y = f / view_pc[..., 2] * view_pc[..., 1] + K_init[1, 2]
        bp_imgpt = torch.stack([bp_imgpt_x, bp_imgpt_y], dim=-1)

        # loss = ((px[px_mask] - bp_imgpt[px_mask]) ** 2).sum(-1).mean()

        loss = criterion(px[px_mask], bp_imgpt[px_mask])

        # if (loss.data - p_loss).abs() < 1e-16:
        #     bConverge = True

        p_loss = loss.data

        n_iter.set_description(f'{message}[Camera Bundle Adjustment {i}/{max_iteration}] loss: {loss.data:2f}')

        loss.backward()
        optimiser.step()

    cam_final.K = torch.tensor([[f, 0, K_init[0, 2]],
                                [0, f, K_init[1, 2]],
                                [0, 0, 1]]).to(device).repeat(nView, 1, 1)

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
def RtPC_f_BA_K(K: torch.tensor, R_init: torch.tensor, t_init: torch.tensor,
                px: torch.tensor, pc_init: torch.tensor,
                px_mask: torch.tensor = None, pc_mask: torch.tensor = None):
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


def Rt_f_BA_K(K: torch.tensor, R_init: torch.tensor, t_init: torch.tensor,
              px: torch.tensor, pc_init: torch.tensor,
              px_mask: torch.tensor = None, pc_mask: torch.tensor = None):
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
