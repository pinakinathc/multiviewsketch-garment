# -*- coding: utf-8 -*-

import numpy as np
import argparse
import trimesh
from matplotlib import cm
from igl import signed_distance


def resize_mesh(mesh):
    mesh = trimesh.Trimesh(mesh.vertices - mesh.centroid, mesh.faces)
    scene = trimesh.Scene(mesh)
    scene = scene.scaled(1.5/scene.scale)
    mesh = scene.geometry[list(scene.geometry.keys())[0]]
    return mesh


def error_surface(gt_mesh, pred_mesh, out_mesh):
    gt_mesh = trimesh.load(gt_mesh)
    pred_mesh = trimesh.load(pred_mesh)

    gt_mesh = resize_mesh(gt_mesh)
    pred_mesh = resize_mesh(pred_mesh)

    sdf = signed_distance(pred_mesh.vertices, gt_mesh.vertices, gt_mesh.faces)[0]
    sdf = ((sdf - sdf.min())/(sdf.max() - sdf.min() + 1e-12))
    cmap = cm.get_cmap('seismic')
    color_val = cmap(sdf)

    to_save = np.concatenate([pred_mesh.vertices, color_val], axis=-1)
    fname = out_mesh

    with open(fname, 'w') as fp:
        for v in to_save:
            fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], v[3], v[4], v[5]))
        for f in pred_mesh.faces+1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute error surface of predicted mesh w.r.t. GT mesh')
    parser.add_argument('--gt_mesh', type=str, help='GT mesh')
    parser.add_argument('--pred_mesh', type=str, help='Pred mesh')
    parser.add_argument('--out_mesh', type=str, help='Output error mesh')
    opts = parser.parse_args()

    error_surface(opts.gt_mesh, opts.pred_mesh, opts.out_mesh)
