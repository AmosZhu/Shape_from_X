"""
Author: dizhong zhu
Date: 10/08/2022
"""

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from GeoUtils_pytorch.Geo3D import (
    Epipolar,
    PnP,
)
from GeoUtils.common import (
    make_homegenous
)
from GeoUtils_pytorch.Geo3D import reconstruction
from GeoUtils_pytorch.Cameras.PerspectiveCamera import backProjection

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

    vis_table = np.zeros(shape=(nView, nMaxFeat), dtype=bool)
    idx_table = np.zeros(shape=(nView, nMaxFeat), dtype=int)

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


def shift_px_to_origin(px, w, h):
    px_x = px[..., 0] - w / 2
    px_y = px[..., 1] - h / 2

    return np.stack([px_x, px_y], axis=-1)


def shift_px_to_center(px, w, h):
    px_x = px[..., 0] + w / 2
    px_y = px[..., 1] + h / 2

    return np.stack([px_x, px_y], axis=-1)


if __name__ == '__main__':
    noofImages = 5

    images, (Ks, Rs, ts) = datahelper.load_data(noofImages)
    # (images, _), Ks, _ = datahelper.load_data('E:/dataset/KITTI', noofImages=noofImages)
    device = 'cpu'

    gt_P = []
    for K, R, t in zip(Ks, Rs, ts):
        K[0, 2] = 0
        K[1, 2] = 0
        K[0, 1] = 0
        P = K @ np.concatenate([R, t[..., None]], axis=-1)
        gt_P.append(P)

    gt_P = np.stack(gt_P)

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
                             'px': np.array([p.pt for p in kp], dtype=np.float32),
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
            px1_shift = shift_px_to_origin(px1, w, h)
            px2_shift = shift_px_to_origin(px2, w, h)

            F, mask = cv2.findFundamentalMat(px1_shift, px2_shift, cv2.FM_8POINT)  # these points are all inliers

            P = Epipolar.P_f_F(torch.from_numpy(F[None, ...]).to(device).float())

            cams['P'] = torch.eye(3, 4)
            cams['P'] = torch.stack([cams['P'], P[0]])

            pxs = torch.from_numpy(np.stack([px1_shift, px2_shift])).float()
            pt3D = reconstruction.triangulateReconstruction(pxs=pxs.to(device), cams=cams['P'].to(device))

            # back projection test
            img_pt_shift = backProjection(P, pt3D[None, ...])[0].detach().cpu().numpy()
            img_pt = shift_px_to_center(img_pt_shift, w, h)
            plot_landmarks(images[i][None, ...], px2[None, ...], img_pt[None, ...])

            scene_points_inlier = np.ones(pt3D.shape[0], dtype=bool)

            scene_points = pt3D.detach().cpu().numpy()  # initial scene points
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(scene_points)
            # o3d.visualization.draw_geometries([pcd])

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
            visible_px_shift = shift_px_to_origin(visible_px, w, h)

            P = PnP.P_f_PnP(pt2D=torch.from_numpy(visible_px_shift[None]).to(device),
                            pt3D=torch.from_numpy(visible_3D_pt[None]).to(device))  # the 3D point was reconstructed by pixel coordinate shift to origins

            # triangulate all
            cams['P'] = torch.cat([cams['P'], P])
            pxs_shift = np.stack([shift_px_to_origin(feature_list[k]['px'][idx_table[k]], w, h) for k in range(i + 1)])
            pxs = np.stack([feature_list[k]['px'][idx_table[k]] for k in range(i + 1)])

            vis_mask = vis_table[:i + 1]
            pt3D = reconstruction.triangulateReconstruction(pxs=torch.from_numpy(pxs_shift).to(device),
                                                            cams=cams['P'],
                                                            mask=torch.from_numpy(vis_mask).to(device))

            scene_points_inlier = np.ones(pt3D.shape[0], dtype=bool)

            # backproject to all views
            img_pt = []
            img_pt_shift = backProjection(cams['P'], pt3D[None, ...]).detach().cpu().numpy()
            for k in range(i + 1):
                # img_pt_shift = backProjection(cams['P'][k], pt3D)
                img_pt.append(shift_px_to_center(img_pt_shift[k], w, h))
            img_pt = np.stack(img_pt)
            # scene_points_inlier &= ~image_outliers(img_pt, h, w)
            plot_landmarks(images[:i + 1], pxs[:, scene_points_inlier], img_pt[:, scene_points_inlier])
            # cv2.imshow('epipolar constraint', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # scene_points=pt_3d
            # scene_points=pt_3d
            # new_scene_points = np.concatenate([scene_points])

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pt3D[scene_points_inlier])
            # o3d.visualization.draw_geometries([pcd])

            scene_points = pt3D.detach().cpu().numpy()
        pass

    H = PnP.euclidian_rectify_f_P(cams['P']).detach().cpu().numpy()

    rectify_pt = make_homegenous(scene_points) @ np.linalg.pinv(H).T
    rectify_pt = rectify_pt[..., :3] / rectify_pt[..., 3:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rectify_pt[scene_points_inlier])
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)
    # scene_points_inlier = np.zeros(pt3D.shape[0], dtype=bool)
    # scene_points_inlier[ind] = True

    o3d.visualization.draw_geometries([pcd])
