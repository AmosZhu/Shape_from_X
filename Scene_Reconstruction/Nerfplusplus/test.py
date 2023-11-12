"""
Author: dizhong zhu
Date: 06/10/2022
"""

from model.NerfppRaysampler import SphereMonteCarloRaysampler
import torch
from test_datas.matajoh.datahelper import matajohDataset
from pytorch3d.utils import cameras_from_opencv_projection


def random_select_f_dataset(dataset):
    # get a random valid sample
    show_idx = int(torch.randint(low=0, high=len(dataset), size=(1,)))
    (K, R, t), (image, mask) = dataset[show_idx]
    K = torch.from_numpy(K)[None, ...]
    R = torch.from_numpy(R)[None, ...]
    t = torch.from_numpy(t)[None, ...]

    image = torch.from_numpy(image)[None, ...]
    mask = torch.from_numpy(mask)[None, ...]

    return (K, R, t), (image, mask)


if __name__ == '__main__':
    n_rays_per_image = 1024
    min_depth = 1e-4
    stratified = True

    device = 0
    image_height, image_width = 400, 400

    val_nerfdataset = matajohDataset(name=f'lego_400.npz', mode='val')
    (K, R, t), (val_images, val_silhouettes) = random_select_f_dataset(val_nerfdataset)

    raysampler_coarse = SphereMonteCarloRaysampler(
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        n_rays_per_image=n_rays_per_image,
        n_pts_per_ray=128,
        min_depth=min_depth,
        stratified_sampling=stratified
    )

    # convert the camera to pytorch3D camera
    sample_cameras = cameras_from_opencv_projection(R=R.to(device), tvec=t.to(device), camera_matrix=K.to(device),
                                                    image_size=torch.tensor([image_height, image_width]).expand(K.shape[0], 2).to(device))

    ray_bundles = raysampler_coarse(sample_cameras)
