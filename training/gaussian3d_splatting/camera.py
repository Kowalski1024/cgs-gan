from typing import NamedTuple
import torch
import numpy as np


class Camera(NamedTuple):
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    image_height: int = 512
    image_width: int = 512


def extract_cameras(camera_to_world, intrinsics, image_size) -> list[Camera]:
    intrinsics = intrinsics * 512
    w2c = torch.inverse(camera_to_world)

    # Extract FoVx and FoVy from intrinsics matrix
    FoVx = 2 * torch.atan(intrinsics[:, 0, 2] / intrinsics[:, 0, 0])
    FoVy = 2 * torch.atan(intrinsics[:, 1, 2] / intrinsics[:, 1, 1])

    world_view_transform = w2c.transpose(-2, -1)
    zfar = 100.0
    znear = 0.01

    # Calculate projection matrix
    projection_matrix = _get_projection_matrices(znear, zfar, FoVx, FoVy).transpose(
        -2, -1
    )

    full_proj_transform = torch.bmm(world_view_transform, projection_matrix)
    camera_center = torch.inverse(world_view_transform)[:, 3, :3]

    cameras = []
    for i in range(camera_to_world.shape[0]):
        cameras.append(
            Camera(
                FoVx[i].item(),
                FoVy[i].item(),
                world_view_transform[i],
                full_proj_transform[i],
                camera_center[i],
                image_size,
                image_size,
            )
        )

    return cameras


def generate_cameras(batch_size: int, device, image_size):
    poses = torch.stack(
        [
            pose_spherical(
                theta=np.random.uniform(low=-180, high=180),
                phi=90 - uniform_circle(low=90, high=180),
                radius=1.25,
            )
            for _ in range(batch_size)
        ],
        dim=0,
    ).to(device)
    poses[:, :3, 1:3] *= -1
    intrinsics = (
        torch.tensor(
            [[525.0, 0.0, 256.0], [0.0, 525.0, 256.0], [0.0, 0.0, 1.0]], device=device
        )
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    return extract_cameras(poses, intrinsics, image_size)


def pose_spherical(theta, phi, radius):
    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w


def uniform_circle(low: float, high: float):
    """Sample a point uniformly from half of a circle.

    The returned point is described by an angle from `low` to `high`. The samples
    are drawn so that the points on the circle would follow a uniform distribution.
    Angles don't follow a uniform distribution.
    """
    if low < 0 or high > 180 or low > high:
        raise ValueError(
            "Angles must be in range [0, 180] and `low` must be smaller than `high`!"
        )

    low = low / 180 * np.pi
    high = high / 180 * np.pi
    sample = np.random.uniform(low=0, high=1)
    sample_radians = np.arccos(np.cos(low) - sample * (np.cos(low) - np.cos(high)))
    return sample_radians / np.pi * 180


def _get_projection_matrices(znear, zfar, fovX, fovY):
    bath_size = fovX.shape[0]
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros((bath_size, 4, 4), device=fovX.device)

    z_sign = 1.0

    P[:, 0, 0] = 2.0 * znear / (right - left)
    P[:, 1, 1] = 2.0 * znear / (top - bottom)
    P[:, 0, 2] = (right + left) / (right - left)
    P[:, 1, 2] = (top + bottom) / (top - bottom)
    P[:, 3, 2] = z_sign
    P[:, 2, 2] = z_sign * zfar / (zfar - znear)
    P[:, 2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def _rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def _rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()