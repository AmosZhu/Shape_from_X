"""
Author: dizhong zhu
Date: 10/08/2022
"""

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from GeoUtils.Geo3D import (
    Epipolar,
    PnP,
)
from GeoUtils.Geo3D import reconstruction
from GeoUtils.Cameras.cameras import backProjection
from GeoUtils_pytorch.Cameras.PerspectiveCamera import (
    ParameterisePerspectiveCamera
)
from GeoUtils_pytorch.Geo3D import bundlAdjustment as BA
import open3d as o3d
from utils.plots import (
    plot_landmarks,
    plot_imageset
)
from utils.plotly_helper import (
    plotly_imageset,
    plotly_pointcloud_and_camera
)

from test_datas.templeRing import datahelper


# from test_datas.KITTI import datahelper


def plot_realsize(img):
    plt.figure(figsize=(int(img.shape[1] / 100), int(img.shape[0] / 100)))


def construct_matching_table(feature_list):
    nView = len(feature_list)

    nMaxFeat = 0
    for i in range(nView):
        nMaxFeat += len(feature_list[i]['row'])

    vis_table = np.zeros(shape=(nView, nMaxFeat), dtype=np.bool)
    idx_table = np.zeros(shape=(nView, nMaxFeat), dtype=np.int)

    for i in range(nView):
        js = np.where(feature_list[i]['row'] != -1)[0]
        idx = feature_list[i]['row'][js]
        vis_table[i, idx] = True
        idx_table[i, idx] = js

        # for j in range(len(feature_list[i]['row'])):
        #     if feature_list[i]['row'][j] != -1:
        #         idx = feature_list[i]['row'][j]
        #         vis_table[i, idx] = True
        #         idx_table[i, idx] = j

    trunk_idx = np.where(np.all(vis_table == False, axis=0))[0][0]
    return vis_table[..., :trunk_idx], idx_table[..., :trunk_idx]


def image_outliers(imgpt, h, w):
    outlier_x = (imgpt[..., 0] < 0) | (imgpt[..., 0] >= w)
    outlier_y = (imgpt[..., 1] < 0) | (imgpt[..., 1] >= h)
    outliers = outlier_x | outlier_y

    return np.any(outliers, 0)


if __name__ == '__main__':
    noofImages = 4

    images, (Ks, Rs, ts) = datahelper.load_data(noofImages)
    # (images, _), Ks, _ = datahelper.load_data('E:/dataset/KITTI', noofImages=noofImages)
    device = 'cpu'

    # selected_images = range(noofImages)
    # images = images[selected_images]
    # Ks = Ks[selected_images]
    # Rs = Rs[selected_images]
    # ts = ts[selected_images]

    noofImages = images.shape[0]
    h, w = images.shape[1], images.shape[2]

    # plotly_imageset(images, nrow=1).show()
    plot_imageset(images)

    SIFT = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    feature_list = []
    for i in range(noofImages):
        kp, des = SIFT.detectAndCompute(images[i], None)
        feature_list.append({'kp': kp,
                             'des': des,
                             'px': np.array([p.pt for p in kp], dtype=np.float),
                             'row': -np.ones(len(kp), dtype=np.int32)})
        # kp, des = FAST.detectAndCompute(images[i], None)

    # So match the between pairs

    # BFMatcher with default params
    bf = cv2.BFMatcher(crossCheck=True)  # set crossCheck to true get 1-1 correspondences

    scene_points = None
    scene_points_inlier = None  # if some scenepoint back project out of the image, then we consider as an outlier
    feature_count = 0

    K = Ks[0]
    K_tensor = torch.from_numpy(K[None]).to(device).float()
    cams = {'K': None, 'R': None, 't': None, 'P': None}

    for i in range(1, noofImages):
        desi = feature_list[i]['des']
        kpi = feature_list[i]['kp']
        imgi = images[i]

        for j in range(i):
            desj = feature_list[j]['des']
            kpj = feature_list[j]['kp']
            imgj = images[j]

            # get an initial match points
            matches = bf.match(desi, desj)
            matchPoints_i = []
            matchPoints_j = []
            # get the match points
            for k, m in enumerate(matches):
                matchPoints_i.append(kpi[m.queryIdx].pt)
                matchPoints_j.append(kpj[m.trainIdx].pt)

            matchPoints_i = np.array(matchPoints_i)
            matchPoints_j = np.array(matchPoints_j)

            # Let's compute fundamental matrix to force epipolar constraint
            bConverge = False
            F, mask = cv2.findFundamentalMat(matchPoints_i, matchPoints_j, cv2.FM_LMEDS)
            epipolar_matches = [match for i, match in enumerate(matches) if mask[i] == 1]
            nInliers = len(epipolar_matches)
            while not bConverge:
                mask = mask.squeeze() == 1
                matchPoints_i = matchPoints_i[mask]
                matchPoints_j = matchPoints_j[mask]
                F, mask = cv2.findFundamentalMat(matchPoints_i, matchPoints_j, cv2.FM_LMEDS)
                epipolar_matches = [match for i, match in enumerate(epipolar_matches) if mask[i] == 1]
                if len(epipolar_matches) == nInliers:
                    bConverge = True
                nInliers = len(epipolar_matches)

            # cv.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatches(imgi, kpi, imgj, kpj, epipolar_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plot_realsize(img3)
            plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
            plt.show()

            if np.sum(mask) < 100:
                continue  # if there are few inliers, ignore this pair

            # epipolar_matches = [match for i, match in enumerate(matches) if mask[i] == 1]

            for m in epipolar_matches:
                idx_i = m.queryIdx
                idx_j = m.trainIdx
                if feature_list[i]['row'][idx_i] == -1 and feature_list[j]['row'][idx_j] == -1:
                    feature_list[i]['row'][idx_i] = feature_count
                    feature_list[j]['row'][idx_j] = feature_count
                    feature_count += 1
                elif feature_list[i]['row'][idx_i] == -1 and feature_list[j]['row'][idx_j] != -1:
                    feature_list[i]['row'][idx_i] = feature_list[j]['row'][idx_j]
                elif feature_list[j]['row'][idx_j] == -1 and feature_list[i]['row'][idx_i] != -1:
                    feature_list[j]['row'][idx_j] = feature_list[i]['row'][idx_i]
                else:
                    continue

        ## once we've done pairwise point matching, we can do incremental reconstruction now
        vis_table, idx_table = construct_matching_table(feature_list[:i + 1])
        if i == 1:  # we compute only by fundamental matrix to build a start point, a more proper way is to choose the one has large baseline and most features between pairs.
            px1 = feature_list[0]['px'][idx_table[0]]
            px2 = feature_list[1]['px'][idx_table[1]]
            F, mask = cv2.findFundamentalMat(px1, px2, cv2.FM_8POINT)  # these points are all inliers

            E = Epipolar.E_f_F(F, K)
            R_est, t_est, pt3D = reconstruction.PoseEstimation_f_F_K(F, px1, px2, K)

            cams['K'] = K
            cams['R'] = np.eye(3)
            cams['t'] = np.zeros(3)
            cams['s'] = 1
            cams['P'] = K @ np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1)

            cams['K'] = np.stack([cams['K'], K])
            cams['R'] = np.stack([cams['R'], R_est])
            cams['t'] = np.stack([cams['t'], t_est])
            cams['s'] = np.stack([cams['s'], 1])
            cams['P'] = np.stack([cams['P'], K @ np.concatenate([R_est, t_est[..., None]], axis=-1)])

            scene_points = pt3D  # initial scene points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scene_points)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1)
            scene_points_inlier = np.zeros(pt3D.shape[0], dtype=np.bool)
            scene_points_inlier[ind] = True
            o3d.visualization.draw_geometries([cl])

            # # let's do bundle adjustment
            # cameras = ParameterisePerspectiveCamera(
            #     intrinsic=torch.from_numpy(cams['K']).float(),
            #     rotation=torch.from_numpy(cams['R']).float(),
            #     translation=torch.from_numpy(cams['t']).float(),
            # ).to(device)
            # cameras_new, pc_new = BA.RtsPC_f_BA_K(cameras,
            #                                       px=torch.from_numpy(np.stack([px1, px2])).float().to(device),
            #                                       pc_init=torch.from_numpy(scene_points).float().to(device),
            #                                       pc_mask=torch.from_numpy(scene_points_inlier).to(device)
            #                                       )
            #
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pc_new[scene_points_inlier])
            # o3d.visualization.draw_geometries([pcd])
            #
            # cams['R'] = cameras_new.R.detach().cpu().numpy()
            # cams['t'] = cameras_new.t.detach().cpu().numpy()
            # cams['P'] = cameras_new.P.detach().cpu().numpy()
            # cams['s'] = cameras_new.s.detach().cpu().numpy()

            # # save the estimated cameras
            # P = K_tensor @ torch.cat([R_new, t_new[..., None]], dim=-1)
            # cams['R'] = R_new.detach().cpu().numpy()
            # cams['t'] = t_new.detach().cpu().numpy()
            # cams['P'] = P.detach().cpu().numpy()

        else:
            # Now we start to use PnP, don't use pairwise estimation, because the projection matrix from fundamental is not unique. The reconstruction point might be in different scale
            # looking for 3D and 2D correspondence for PnP
            n3Dpoints = len(scene_points)
            visible_3D_mask = vis_table[i, :n3Dpoints]  # we are now in ith view, we only take the 3D points visible by this view
            if scene_points_inlier is not None:
                visible_3D_mask = visible_3D_mask & scene_points_inlier
            visible_3D_pt = scene_points[visible_3D_mask]

            # now take the feature only with visible 3D points
            trunk_idx = idx_table[i, :n3Dpoints][visible_3D_mask]  # because the len(feature_table)>=len(scene_points)
            indices_i = trunk_idx[trunk_idx != -1]
            visible_px = feature_list[i]['px'][indices_i]

            P = PnP.P_f_PnP(visible_px, visible_3D_pt)
            R_est1, t_est1, s_est1 = PnP.Rts_f_P_K(P, K)
            R_est, t_est, s_est = PnP.Rts_f_PnP_K(visible_px, visible_3D_pt, K)
            # P_rect = K @ np.concatenate([R_est, t_est[..., None]], axis=-1)

            # let's do bundle adjustment
            cameras = ParameterisePerspectiveCamera(
                intrinsic=torch.from_numpy(K[None]).float(),
                rotation=torch.from_numpy(R_est[None]).float(),
                translation=torch.from_numpy(t_est[None]).float(),
                scale=torch.from_numpy(s_est[None]).float(),
            ).to(device)
            cameras_new = BA.Rts_f_BA_K(cameras=cameras,
                                        px=torch.from_numpy(visible_px[None]).float().to(device),
                                        pc_init=torch.from_numpy(visible_3D_pt).float().to(device),
                                        )
            R_new = cameras_new.R.detach().cpu().numpy()[0]
            t_new = cameras_new.t.detach().cpu().numpy()[0]
            s_new = cameras_new.s.detach().cpu().numpy()[0]
            P_new_scale = cameras_new.P.detach().cpu().numpy()[0]
            P_new = K @ np.concatenate([R_new, t_new[..., None]], axis=-1)

            # img_pt = backProjection(P, visible_3D_pt)
            # img = plot_landmarks(images[i][None, ...], visible_px[None, ...], img_pt[None, ...])
            # plt.imshow(img)
            # plt.show()
            #
            # img_pt = backProjection(P_rect, visible_3D_pt)
            # img = plot_landmarks(images[i][None, ...], visible_px[None, ...], img_pt[None, ...])
            # plt.imshow(img)
            # plt.show()
            #
            # img_pt = backProjection(P_new_scale, visible_3D_pt)
            # img = plot_landmarks(images[i][None, ...], visible_px[None, ...], img_pt[None, ...])
            # plt.imshow(img)
            # plt.show()

            cams['K'] = np.concatenate([cams['K'], K[None, ...]])
            cams['R'] = np.concatenate([cams['R'], R_new[None, ...]])
            cams['t'] = np.concatenate([cams['t'], t_new[None, ...]])
            cams['s'] = np.concatenate([cams['s'], s_new[None, ...]])
            cams['P'] = np.concatenate([cams['P'], P_new_scale[None, ...]])

            pxs = np.stack([feature_list[k]['px'][idx_table[k]] for k in range(i + 1)])
            # pxs = np.stack(pxs)

            # for k in range(i + 1):
            #     pxs.append(feature_list[k]['px'][idx_table[k]])

            vis_mask = vis_table[:i + 1]

            pt3D = reconstruction.triangulateReconstruction(pxs=pxs, cams=cams['P'], mask=vis_mask)

            # let's do bundle adjustment
            scene_points_inlier_BA = np.zeros(pt3D.shape[0], dtype=np.bool)
            scene_points_inlier_BA[:n3Dpoints] = scene_points_inlier
            cameras = ParameterisePerspectiveCamera(
                intrinsic=torch.from_numpy(cams['K']).float(),
                rotation=torch.from_numpy(cams['R']).float(),
                translation=torch.from_numpy(cams['t']).float(),
                scale=torch.from_numpy(cams['s']).float()
            ).to(device)
            cameras_new, pt3D = BA.RtsPC_f_BA_K(cameras,
                                                px=torch.from_numpy(pxs).float().to(device),
                                                pc_init=torch.from_numpy(pt3D).float().to(device),
                                                px_mask=torch.from_numpy(vis_mask).to(device),
                                                pc_mask=torch.from_numpy(scene_points_inlier_BA).to(device)
                                                )

            # save the estimated cameras
            pt3D = pt3D.detach().cpu().numpy()
            cams['R'] = cameras_new.R.detach().cpu().numpy()
            cams['t'] = cameras_new.t.detach().cpu().numpy()
            cams['P'] = cameras_new.P.detach().cpu().numpy()
            cams['s'] = cameras_new.s.detach().cpu().numpy()

            # remove outliers
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt3D)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1)
            scene_points_inlier = np.zeros(pt3D.shape[0], dtype=np.bool)
            scene_points_inlier[ind] = True

            # backproject to all views
            img_pt = []
            for k in range(i + 1):
                img_pt.append(backProjection(cams['P'][k], pt3D))
            img_pt = np.stack(img_pt)
            scene_points_inlier &= ~image_outliers(img_pt, h, w)
            img = plot_landmarks(images[:i + 1], pxs[:, scene_points_inlier], img_pt[:, scene_points_inlier])
            cv2.imshow('epipolar constraint', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # scene_points=pt_3d
            # scene_points=pt_3d
            # new_scene_points = np.concatenate([scene_points])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pt3D[scene_points_inlier])
            o3d.visualization.draw_geometries([pcd])

            scene_points = pt3D
        pass

