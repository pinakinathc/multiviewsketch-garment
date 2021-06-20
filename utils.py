# -*- coding: utf-8 -*-
# author: pinakinathc.me

import numpy as np
import torch
from skimage import measure
from sdf import create_grid, eval_grid_octree, eval_grid

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def projection(points, calib, projection_mode='orthogonal'):
	func = orthogonal if projection_mode == 'orthogonal' else perspective
	return func(points, calib)


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


def visualise_NDF(ndf):
    ''' Visualise NDF '''
    reso = ndf.shape[0]
    assert ndf.shape == (reso, reso, reso), 'NDF shape not equal to 256x256x256'
    import matplotlib.pyplot as plt

    print ('visualising NDF')
    for z in range(0, reso, max(reso//10, 1)):
        uv = ndf[:,::-1,z].T
        plt.imshow(uv)
        # plt.show()
        plt.savefig(str(z)+'.png')
        print ('saved ', z)    


def gen_mesh(opt, model, cuda, data, save_path, use_octree=True, num_samples=10000):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    b_min = data['b_min']
    b_max = data['b_max']
    resolution = 32

    label_tensor = data['labels'].to(device=cuda)
    print (label_tensor.shape)
    gt_ndf = label_tensor.detach().cpu().numpy().reshape(resolution, resolution, resolution)
    # visualise_NDF(label_tensor.detach().cpu().numpy().reshape(resolution, resolution, resolution))

    coords, mat = create_grid(resolution, resolution, resolution,
            b_min, b_max, transform=None)

    # define lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        projected_sample_tensor = projection(samples, calib_tensor).permute(0, 2, 1)
        print ('samples shape: ', samples.shape, ', projected_sample_tensor shape: ', projected_sample_tensor.shape)
        pred, _ = model(image_tensor, projected_sample_tensor)
        return pred.detach().cpu().numpy()

    # evaluate grid
    if use_octree and False:
        ndf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        ndf = eval_grid(coords, eval_func, num_samples=num_samples)

    from dataloader import save_samples_truncted_prob
    coords = coords.reshape(3, -1).T
    print ('shape of coords: {}, ndf: {}'.format(coords.shape, ndf.shape))
    print ('ndf min: {}, max: {}, mean: {}'.format(np.min(ndf), np.max(ndf), np.mean(ndf)))
    # save_samples_truncted_prob('testing.ply', coords, ndf)

    visualise_NDF(ndf)
    print ('check pred')

    # do marching cubes
    # try:
    verts, faces, normals, values = measure.marching_cubes_lewiner(ndf, 0.1)
    # mat = calib_tensor[0].detach().cpu().numpy() # TODO remove, required for GT only
    print ('initial verts shape: {}, multiplied: {}, mat: {}'.format(verts.shape, np.matmul(mat[:3, :3], verts.T).shape, mat[:, 3:4].shape))
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    print ('shape of calib: {}, mat: {}, verts: {}, save_path: {}'.format(calib_tensor.shape, mat.shape, verts.shape, save_path))

    # Smoothing
    import trimesh
    new_mesh = trimesh.Trimesh(vertices=verts.T, faces=faces)
    smoothed = trimesh.smoothing.filter_laplacian(new_mesh,lamb=0.5)
    save_obj_mesh(save_path, smoothed.vertices, smoothed.faces)
    # except:
    #     print ('error cannot marching cubes')
    #     return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    print ('shape of vertices: {}, faces: {}'.format(
        verts.shape, faces.shape))
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()
