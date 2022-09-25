"""
Author: dizhong zhu
Date: 08/04/2022

A great explanation would be here:
https://github.com/matajoh/fourier_feature_nets
"""
import inspect
import os
import cv2
from GeoUtils_pytorch.Cameras.PerspectiveCamera import PerspectiveCamera

from pytorch3d.utils import cameras_from_opencv_projection
from torch.nn.parallel import DistributedDataParallel as DDP

from model.NeuralRadianceField import (
    BasicNeuralRadianceField,
    FourierNerf,
    Nerf
)
from test_datas.matajoh.datahelper import matajohDataset
from torch.utils.data import (
    DataLoader,
)
from torch.utils.tensorboard import SummaryWriter

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
from torch.utils.data.distributed import DistributedSampler

from model.NerfRaymarcher import NerfRaymarcher
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from pytorch3d.transforms import so3_exp_map
from utils.image_helper import (
    make_video
)
from GeoUtils_pytorch.Geo3D.Rotation import (
    rotx
)
from utils.plotly_helper import plotly_pointcloud_and_camera

# from nerf_utils import (
#     make_video,
#     generate_rotating_nerf,
#     generate_by_camera
# )
#
# from utils.display_camera import plot_cameras
import visdom

mse2psnr = lambda x: -10. * np.log(x + 1e-10) / np.log(10.)


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


@torch.no_grad()
def tensorboard_add_images(
        writer: SummaryWriter,
        neural_radiance_field, camera, renderer_grid,
        target_image, target_silhouette,
        epoch=0,
        max_depth=1):
    # Render using the grid renderer and the
    # batched_forward function of neural_radiance_field.
    rendered_image_silhouette, _ = renderer_grid(
        cameras=camera,
        volumetric_function=neural_radiance_field.batch_forward,
        max_depth=max_depth
    )
    # Split the rendering result to a silhouette render
    # and the image render.

    rendered_image, rendered_silhouette, rendered_depth = (
        rendered_image_silhouette[0].split([3, 1, 1], dim=-1)
    )

    clamp = lambda x: torch.clamp(x, 0.0, 1.0)
    writer.add_image(f'epoch_{epoch}/GT_RGB', clamp(target_image), epoch)
    writer.add_image(f'epoch_{epoch}/GT_silhouette', clamp(target_silhouette), epoch)

    writer.add_image(f'epoch_{epoch}/rendered_RGB', clamp(rendered_image.permute(2, 0, 1)), epoch)
    writer.add_image(f'epoch_{epoch}/rendered_silhouette', clamp(rendered_silhouette.permute(2, 0, 1)), epoch)
    writer.add_image(f'epoch_{epoch}/rendered_depth', rendered_depth.permute(2, 0, 1) / max_depth, epoch)


def generate_rotating_nerf(nerf_model, renderer: ImplicitRenderer, K, image_size, max_depth=1.0, n_frames=50, device=0):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(0.0, 2.0 * 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)

    Rx = rotx(torch.Tensor([-30] * n_frames)).to(device)

    Rs = Rx @ Rs

    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 3.5

    Ks = K.expand(n_frames, 3, 3).to(device)

    print('Generating rotating nerf videos ...')

    sample_cameras = cameras_from_opencv_projection(R=Rs, tvec=Ts, camera_matrix=Ks,
                                                    image_size=torch.Tensor([image_size[0], image_size[1]]).expand(n_frames, 2).to(device))

    images = []
    for i in tqdm(range(n_frames)):
        rendered_images, sampled_rays = renderer(
            cameras=sample_cameras[i],
            volumetric_function=nerf_model.batch_forward,
            max_depth=max_depth
        )

        img_alpha, img_rgb = rendered_images[..., 3], rendered_images[..., :3]
        img_cpu = np.uint8(img_rgb.cpu().numpy()[0] * 255.0)
        img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
        img_cpu = cv2.flip(img_cpu, 0)
        images.append(img_cpu)

    images = np.stack(images)
    return images


def show_full_render(
        neural_radiance_field, camera, renderer_grid,
        target_image, target_silhouette,
        loss_history_color, loss_history_sil,
        epoch=0,
        max_depth=1
):
    """
    This is a helper function for visualizing the
    intermediate results of the learning.

    Since the `NeuralRadianceField` suffers from
    a large memory footprint, which does not let us
    render the full image grid in a single forward pass,
    we utilize the `NeuralRadianceField.batched_forward`
    function in combination with disabling the gradient caching.
    This chunks the set of emitted rays to batches and
    evaluates the implicit function on one batch at a time
    to prevent GPU memory overflow.
    """

    # Prevent gradient caching.
    with torch.no_grad():
        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        rendered_image_silhouette, _ = renderer_grid(
            cameras=camera,
            volumetric_function=neural_radiance_field.batch_forward,
        )
        # Split the rendering result to a silhouette render
        # and the image render.

        rendered_image, rendered_silhouette, rendered_depth = (
            rendered_image_silhouette[0].split([3, 1, 1], dim=-1)
        )

    # Generate plots.
    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    ax = ax.ravel()
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    ax[0].plot(list(range(len(loss_history_color))), loss_history_color, linewidth=1)
    ax[1].imshow(clamp_and_detach(rendered_image))
    ax[2].imshow(clamp_and_detach(rendered_silhouette[..., 0]))
    ax[3].imshow(rendered_depth.detach().cpu().numpy() / max_depth, cmap='gray')

    ax[4].plot(list(range(len(loss_history_sil))), loss_history_sil, linewidth=1)
    ax[5].imshow(clamp_and_detach(target_image.permute(1, 2, 0)))
    ax[6].imshow(clamp_and_detach(target_silhouette[0]))
    for ax_, title_ in zip(
            ax,
            (
                    "loss color", "rendered image", "rendered silhouette", "rendered depth",
                    "loss silhouette", "target image", "target silhouette",
            )
    ):
        if not title_.startswith('loss'):
            ax_.grid("off")
            ax_.axis("off")
        ax_.set_title(title_)
    fig.canvas.draw()
    fig.show()

    return fig


def sample_images_from_rays(images, sampled_xy):
    ba = images.shape[0]
    dim = images.shape[1]

    # target_img = images.permute(0, 3, 1, 2)
    images_sampled = torch.nn.functional.grid_sample(images, -sampled_xy.view(ba, -1, 1, 2), align_corners=True)

    return images_sampled.squeeze(-1).transpose(1, 2)  # images_sampled.view(ba, sampled_xy.shape[1], dim)


def train_nerf(device, world_size, epochs=100, learning_rate=1e-3, save_epochs=10, batch_size=10, model_sel='mlp', object='lego', output_dir='output'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12310'

    print('rank: {}'.format(device))
    output_folder = os.path.join(output_dir, model_sel, object)
    os.makedirs(output_folder, exist_ok=True)

    # initialize the process group
    # have to set nccl and torch.cuda.set_device to make avoid gradient compute in dvice 0
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=device, world_size=world_size)

    torch.cuda.set_device(device)

    camera_sample_size = batch_size
    # Set the seed for reproducibility
    torch.manual_seed(device * 100)

    trained_nerfdataset = matajohDataset(name=f'{object}_400.npz')
    val_nerfdataset = matajohDataset(name=f'{object}_400.npz', mode='val')
    # cam_center = trained_nerfdataset.get_camera_matrices()[:, :3, 3]

    min_depth, max_depth = 0.1, 4.0  # trained_nerfdataset.max_depth()  #

    # display cameras
    if device == 0:
        writer = SummaryWriter(f'{output_folder}/tensorboard-logs')
        fig = plotly_pointcloud_and_camera(cameras=[{'name': 'cameras', 'data': trained_nerfdataset.get_camera_matrices()}])
        fig.write_html(f'{output_folder}/cameras.html')

    randomSampler = DistributedSampler(trained_nerfdataset, num_replicas=world_size, rank=device, shuffle=True)

    nerfdataloader = DataLoader(trained_nerfdataset, batch_size=camera_sample_size, sampler=randomSampler, num_workers=0, shuffle=False)
    print(f'Generated {len(trained_nerfdataset)} images/silhouettes/cameras.')

    image_height, image_width = trained_nerfdataset.height, trained_nerfdataset.width

    ##########################################################################################
    #   Setup rendering tools for Nerf
    #   We use the architecture from pytorch3D, while it's very nice and easy to customise
    #   Suppose to contain:
    #   1. A ray marching algorihm
    #   2. A ray sampler
    #   3. A renderer
    ##########################################################################################
    raymarcher = NerfRaymarcher()
    # Dense ray sampler will be use for rendering
    raysampler_grid = NDCMultinomialRaysampler(
        image_width=image_width,
        image_height=image_height,
        n_pts_per_ray=150,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid, raymarcher=raymarcher,
    )

    # MonteCarlo raysampler will take random rays from images, it will be using for training
    raysampler_mc = MonteCarloRaysampler(
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        n_rays_per_image=1024,
        n_pts_per_ray=128,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    renderer_mc = ImplicitRenderer(
        raysampler=raysampler_mc, raymarcher=raymarcher,
    )

    if model_sel == 'mlp':
        nerfModel = BasicNeuralRadianceField(num_features=256, num_layers=3).to(device)
    elif model_sel == 'fourier':
        nerfModel = FourierNerf(num_features=256, num_layers=2, num_basis=64, max_log_scale=6).to(device)
    elif model_sel == 'nerf':
        nerfModel = Nerf(num_features=256, pos_basis=60, view_basis=24).to(device)
    else:
        raise ValueError(f'{model_sel} is not a valid model selection.')

    checkpoint_path = f'{output_dir}/{model_sel}/{object}/nerf_epoch_latest.pt'
    if os.path.exists(checkpoint_path):
        print('load last checkpoints')
        map_location = torch.device('cuda:{}'.format(device))
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        nerfModel.load_state_dict(checkpoint['state'])

    ddp_nerfModel = DDP(nerfModel, device_ids=[device], output_device=device, broadcast_buffers=True)
    # ddp_nerfModel = nerfModel
    optimiser = torch.optim.Adam(ddp_nerfModel.parameters(), lr=learning_rate)

    # Init the loss history buffers.
    loss_history_color, loss_history_sil = [], []

    for epoch in range(epochs):
        tqdm_data = tqdm(nerfdataloader)
        nerfModel.train()

        if epoch > 0.7 * epochs:
            optimiser = torch.optim.Adam(ddp_nerfModel.parameters(), lr=learning_rate / 10)

        for i, sample_data in enumerate(tqdm_data):
            optimiser.zero_grad()

            (K, R, t), (target_images, target_silhouettes) = sample_data

            # convert the camera to pytorch3D camera
            sample_cameras = cameras_from_opencv_projection(R=R.to(device), tvec=t.to(device), camera_matrix=K.to(device),
                                                            image_size=torch.tensor([image_height, image_width]).expand(K.shape[0], 2).to(device))

            # Evaluate the nerf model.
            sample_images, sampled_rays = renderer_mc(
                cameras=sample_cameras,
                volumetric_function=ddp_nerfModel,
                stratified_sampling=True,
                max_depth=max_depth,
            )

            sample_img_rgb = sample_images[..., :3]
            sample_img_alpha = sample_images[..., 3]

            sample_target_img_rgb = sample_images_from_rays(target_images.to(device), sampled_rays.xys)
            sample_target_img_alhpa = sample_images_from_rays(target_silhouettes.to(device), sampled_rays.xys).squeeze(-1)

            loss_rgb = (sample_img_rgb - sample_target_img_rgb).square().mean()
            loss_alpha = (sample_img_alpha - sample_target_img_alhpa).square().mean()
            psnr_score = mse2psnr(loss_rgb.item())

            tqdm_data.set_description(f'[epoch {epoch}/{epochs}] rgb loss: {loss_rgb.data:.4e}, alpha loss: {loss_alpha.data:.4e}, psnr: {psnr_score:.4f}')

            loss = loss_rgb

            # Log the loss history.
            loss_history_color.append(float(loss_rgb))
            loss_history_sil.append(float(loss_alpha))

            loss.backward()
            optimiser.step()

        dist.barrier()

        # record the loss into tensorboard
        if device == 0:
            writer.add_scalar('loss/rgb', loss_rgb, epoch)
            writer.add_scalar('loss/pnsr', mse2psnr(loss_rgb.item()), epoch)

        # Visualize the renders every 100 iterations.
        if device == 0 and epoch % save_epochs == 0:
            with torch.no_grad():
                # for show_idx in range(len(sample_cameras)):
                (val_K, val_R, val_t), (val_images, val_silhouettes) = random_select_f_dataset(val_nerfdataset)
                val_cameras = cameras_from_opencv_projection(R=val_R.to(device), tvec=val_t.to(device), camera_matrix=val_K.to(device),
                                                             image_size=torch.tensor([image_height, image_width]).expand(val_K.shape[0], 2).to(device))

                tensorboard_add_images(writer,
                                       nerfModel,
                                       val_cameras,
                                       renderer_grid,
                                       val_images[0],
                                       val_silhouettes[0],
                                       epoch=epoch,
                                       max_depth=max_depth)

                # f = show_full_render(
                #     nerfModel,
                #     val_cameras,
                #     renderer_grid,
                #     val_images[0],
                #     val_silhouettes[0],
                #     loss_history_color,
                #     loss_history_sil,
                #     epoch=epoch,
                #     max_depth=max_depth
                # )
                #
                # # Save the renders and checkpoints

                #
                # f.savefig(os.path.join(output_folder, f'{epoch:05d}.png'))

                checkpoint_info = {
                    'state': nerfModel.state_dict(),
                    'K': K[0].detach().cpu().numpy(),
                    'image_size': [image_height, image_width],
                    'max_depth': max_depth,
                    'model': model_sel,
                }
                torch.save(checkpoint_info, f'{output_folder}/nerf_epoch_{epoch}.pt')
                torch.save(checkpoint_info, f'{output_folder}/nerf_epoch_latest.pt')

                # rotating_volume_frames = generate_rotating_nerf(nerfModel, renderer_grid, K=K[0],
                #                                                 image_size=[image_height, image_width],
                #                                                 max_depth=max_depth,
                #                                                 n_frames=7 * 4,
                #                                                 device=device)
                # make_video(rotating_volume_frames, f'{output_folder}/rotating_nerf_epoch_{epoch}.mp4')

    dist.destroy_process_group()


def evaluate_nerf(checkpoint_path, video_path):
    device = torch.device('cuda:0')

    # map_location = torch.device('cuda:{}'.format(device))
    # trained_nerfdataset = nerfdata(path=f'data/{object}_400.npz')
    # trained_nerfdataset.save_all_images('test')

    # (K, R, t), (target_images, target_silhouettes) = trained_nerfdataset.get_all_data()  # we need the intrinsic for rendering

    # init_cam = torch.cat([R[0], t[0][..., None]], dim=-1)
    #
    checkpoint = torch.load(checkpoint_path, map_location=device)
    K = torch.from_numpy(checkpoint['K']).to(device)
    image_height, image_width = checkpoint['image_size']
    max_depth = checkpoint['max_depth']

    raysampler_grid = NDCMultinomialRaysampler(
        image_width=image_width,
        image_height=image_height,
        n_pts_per_ray=150,
        min_depth=0.1,
        max_depth=4.0  # trained_nerfdataset.max_depth(),
    )

    # raymarcher = EmissionAbsorptionRaymarcher()
    raymarcher = NerfRaymarcher()

    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid, raymarcher=raymarcher,
    )

    model_sel = checkpoint['model']
    if model_sel == 'mlp':
        nerfModel = BasicNeuralRadianceField(num_features=256, num_layers=3).to(device)
    elif model_sel == 'fourier':
        nerfModel = FourierNerf(num_features=256, num_layers=2, num_basis=64, max_log_scale=6).to(device)
    elif model_sel == 'nerf':
        nerfModel = Nerf(num_features=256, pos_basis=60, view_basis=24).to(device)
    else:
        raise ValueError(f'{model_sel} is not a valid model selection.')

    nerfModel.load_state_dict(checkpoint['state'])

    nerfModel.eval()

    n_sample = 20

    with torch.no_grad():
        # frames = generate_by_camera(nerfModel, renderer_grid, K=K[:n_sample].to(device), R=R[:n_sample].to(device), t=t[:n_sample].to(device), image_size=[image_height, image_width])
        # path = 'nueral_radiance/renderer_image'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        #
        # for i, img in enumerate(frames):
        #     fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        #     ax = ax.ravel()
        #     clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
        #     ax[0].imshow(clamp_and_detach(target_images[i]))
        #     ax[1].imshow(img)
        #
        #     fig.savefig(f'{path}/{i}.png')
        #
        #     # cv2.imwrite(f'{path}/{i}.png', img)

        rotating_volume_frames = generate_rotating_nerf(nerfModel, renderer_grid, K=K, max_depth=max_depth, image_size=[image_height, image_width], n_frames=7 * 4)
        make_video(rotating_volume_frames, f'{video_path}/rotating_nerf.mp4')
