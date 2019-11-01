from collections import OrderedDict
from functools import partial
import json
from pathlib import Path
from shutil import copy

from scipy.io import savemat
import cv2

import numpy as np
from menpo.base import LazyList
import menpo.io as mio
from menpo.io.output.base import _validate_filepath
import menpo3d.io as m3io
from menpo.image.base import normalize_pixels_range
from menpo3d.io.output.mesh import obj_exporter
from menpo.shape import ColouredTriMesh
export_pickle = partial(mio.export_pickle, protocol=4)
import_pickle = partial(mio.import_pickle, encoding='latin1')


def ensure_exists(p):
    if not p.is_dir():
        p.mkdir(parents=True)


def initialize_root(root):
    ensure_exists(root)

    ensure_exists(root / 'shape_nicp')
    ensure_exists(root / 'problematic')

    ensure_exists(root / 'visualizations' / 'landmarks')
    ensure_exists(root / 'visualizations' / 'shape_nicp')
    ensure_exists(root / 'visualizations' / 'pruning')


def import_mesh(path):
    if path.suffix == '.pkl' or path.suffix == '.gz':
        mesh = import_pickle(path)
    else:
        mesh = m3io.import_mesh(path)
    if mesh.texture.pixels.dtype != np.float64:
        mesh.texture.pixels = normalize_pixels_range(mesh.texture.pixels)
    return mesh

def import_mesh_as_color_trimesh(path):
    img = None
    h, w = 0,0
    
    mean_points = []
    trilist = []
    vt_colors = []
    map3dto2d = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("mtllib"):
                # Get texture
                mtl_path = path.parent / line.strip().split()[-1]
                mtl = open(mtl_path, "r")
                for l in mtl:
                    if l.startswith("map_Kd"):
                        img_path = str(mtl_path.parent / l.strip().split()[-1])
                        print("Read image at", img_path)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h,w, _ = img.shape
                mtl.close()
            if line.startswith("v "):
                tokens = line.strip().split()
                x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
                mean_points.append([x,y,z])
            if line.startswith("vt "):
                tokens = line.strip().split()
                x, y = float(tokens[1]), float(tokens[2])
                vt_colors.append(img[int((1-y)*h), int(x*w)].tolist())
            if line.startswith("f "):
                tokens = line.strip().split()
                tripoints = []
                for comp in tokens[1:]:
                    pts = comp.split("/")
                    tripoints.append(int(pts[0]) - 1)
                    map3dto2d[int(pts[0])-1] = int(pts[1])-1
                trilist.append(tripoints[:3])
    mean_colour = [None] * len(mean_points)

    for k, v in map3dto2d.items():
        mean_colour[k] = vt_colors[v]
    mean_points = np.array(mean_points)
    trilist = np.array(trilist)
    mean_colour = np.array(mean_colour) / 255
    return ColouredTriMesh(mean_points, trilist=trilist, colours=mean_colour)
            

def path_settings(r):
    return r / 'settings.json'


def _save_settings_to_path(settings, path, overwrite=False):
    path = _validate_filepath(path, overwrite)
    settings_json = settings.copy()
    settings_json['ids_to_paths'] = {id_: str(path) for id_, path in
                                     settings['ids_to_paths'].items()}
    with open(str(path), 'wt') as f:
        json.dump(settings_json, f, sort_keys=True, indent='    ')


def _load_settings_for_path(path):
    with open(str(path), 'rt') as f:
        settings = json.load(f)
    settings['ids_to_paths'] = OrderedDict(sorted(
        [(id_, Path(path)) for id_, path in settings['ids_to_paths'].items()],
        key=lambda x: x[0]))
    return settings


def import_settings(r):
    return _load_settings_for_path(path_settings(r))


def export_settings(r, settings, overwrite=False):
    _save_settings_to_path(settings, path_settings(r), overwrite=overwrite)


def path_shape_nicp(r, id_):
    return r / 'shape_nicp' / '{}.pkl'.format(id_)


def _load_shape_nicp_for_path(path):
    from .data import load_template  # circular import
    mesh = load_template().from_vector(import_pickle(path))
    mesh.path = path
    return mesh


def import_shape_nicp(r, id_):
    return _load_shape_nicp_for_path(path_shape_nicp(r, id_))


def export_shape_nicp(r, id_, mesh):
    export_pickle(mesh.as_vector(), path_shape_nicp(r, id_), overwrite=True)


def paths_shape_nicp(r):
    return sorted(list(mio.pickle_paths(str(path_shape_nicp(r, '*')))))


def shape_nicp_ids(r):
    return [p.stem for p in paths_shape_nicp(r)]


def shape_nicps(r):
    return LazyList.init_from_iterable(paths_shape_nicp(r),
                                       f=_load_shape_nicp_for_path)


def path_initial_shape_model(r):
    return r / 'initial_shape_model.pkl'


def import_initial_shape_model(r):
    return import_pickle(path_initial_shape_model(r))


def export_initial_shape_model(r, model):
    export_pickle(model, path_initial_shape_model(r))


def path_shape_model(r):
    return r / 'shape_model.mat'


def path_shape_model_cropped(r):
    return r / 'shape_model_cropped.mat'


def export_lsfm_model(pca, n_training_samples, path, extra_dict=None):
    if extra_dict is None:
        extra_dict = {}
    mdict = {
        'components': pca.components.T,
        'eigenvalues': pca.eigenvalues,
        'cumulative_explained_variance': pca.eigenvalues_cumulative_ratio(),
        'mean': pca.mean_vector,
        'n_training_samples': n_training_samples,
        'trilist': pca.mean().trilist
    }
    for k, v in extra_dict.items():
        mdict[k] = v

    savemat(str(path), mdict)

    # if name.endswith('_tri'):
    #     masking_info = mio.import_pickle(
    #         model_path.parent.parent / 'radial_mask_tri.pkl')
    #     mdict['map_crop_to_full'] = masking_info['map_cropped_to_full']
    #     mdict['map_full_to_cropped'] = masking_info['map_full_to_cropped']
    #     name = name.split('_tri')[0]


def path_problematic(r, id_):
    return r / 'problematic' / '{}.txt'.format(id_)


def export_problematic(r, id_, msg):
    with open(str(path_problematic(r, id_)), 'wt') as f:
        f.write(msg)


# ---------------------------- VISUALIZATION IO ------------------------------ #
def path_landmark_visualization(r, id_):
    return r / 'visualizations' / 'landmarks' / '{}.png'.format(id_)


def export_landmark_visualization(r, id_, img):
    mio.export_image(img, path_landmark_visualization(r, id_), overwrite=True)


def path_shape_nicp_visualization(r, id_):
    return r / 'visualizations' / 'shape_nicp' / '{}.png'.format(id_)


def export_shape_nicp_visualization(r, id_, img):
    mio.export_image(img, path_shape_nicp_visualization(r, id_), overwrite=True)


def path_pruning_visualization(r, id_, rank, w_norm, width):
    return (r / 'visualizations' / 'pruning' /
            '{rank:0{width}} - {w_norm:.5f} ({id_}).png'.format(
                rank=rank, width=width, w_norm=w_norm, id_=id_))


def export_pruning_visualization(r, id_, rank, w_norm, n_meshes=10000):
    width = len(str(n_meshes))
    nicp_vis_path = path_shape_nicp_visualization(r, id_)
    prune_result_path = path_pruning_visualization(r, id_, rank, w_norm,
                                                   width=width)
    copy(str(nicp_vis_path), str(prune_result_path))
