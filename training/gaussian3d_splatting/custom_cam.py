import math
import torch
from typing import NamedTuple


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


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

class CustomCam:
    def __init__(self, width, height, fovy, fovx, extr, znear=0.01, zfar=100):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self.world_view_transform = extr.T.inverse()
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform[3, :3]


class Camera(NamedTuple):
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    image_height: int = 256
    image_width: int = 256


def extract_cameras(camera_to_world, intrinsics, image_size) -> list[Camera]:
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