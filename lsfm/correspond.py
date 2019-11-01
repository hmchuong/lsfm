from functools import lru_cache
import numpy as np
from .nicp import non_rigid_icp
from .data import load_template
from scipy.sparse import csr_matrix


def smootherstep(x, x_min, x_max):
    y = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 6 * (y ** 5) - 15 * (y ** 4) + 10 * (y ** 3)


def generate_data_weights(template, nosetip, r_mid=1.05, r_width=0.3,
                          y_pen=1.4, w_inner=1, w_outer=0):
    r_min = r_mid - (r_width / 2)
    r_max = r_mid + (r_width / 2)
    w_range = w_inner - w_outer
    x = np.sqrt(np.sum((template.points - nosetip.points) ** 2 *
                       np.array([1, y_pen, 1]), axis=1))
    return ((1 - smootherstep(x, r_min, r_max))[:, None] * w_range + w_outer).T


def generate_data_weights_per_iter(template, nosetip, r_width, w_min_iter,
                                   w_max_iter, r_mid=10.5, y_pen=1.4):
    # Change in the data term follows the same pattern that is used for the
    # stiffness weights
    stiffness_weights = np.array([50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2])
    s_iter_range = stiffness_weights[0] - stiffness_weights[-1]
    w_iter_range = w_max_iter - w_min_iter
    m = w_iter_range / s_iter_range
    c = w_max_iter - m * stiffness_weights[0]
    w_outer = m * stiffness_weights + c
    w_inner = 1
    return generate_data_weights(template, nosetip, w_inner=w_inner,
                                 w_outer=w_outer, r_width=r_width, r_mid=r_mid,
                                 y_pen=y_pen)


@lru_cache()
def data_weights(source_mesh):
    w_max_iter = 0.5
    w_min_iter = 0.0
    r_width = 0.5 * 0.84716526594210229
    r_mid = 0.95 * 0.84716526594210229
    y_pen = 1.7
    return generate_data_weights_per_iter(source_mesh,
                                          source_mesh.landmarks['nosetip'],
                                          r_width=r_width,
                                          r_mid=r_mid,
                                          w_min_iter=w_min_iter,
                                          w_max_iter=w_max_iter,
                                          y_pen=y_pen
                                          )


def correspond_mesh(mesh, mask=None, verbose=False):
    template = load_template().copy()
    if mask is not None:
        template.landmarks['__lsfm_masked'] = template.landmarks[
            '__lsfm'].from_mask(mask)
        group = '__lsfm_masked'
        if verbose:
            n_bad_lms = (~mask).sum()
            if not n_bad_lms == 0:
                print('occlusion mask provided with {} False values - '
                      'omitting these landmarks from NICP '
                      'calculation'.format(n_bad_lms))
    else:
        group = '__lsfm'
    aligned = non_rigid_icp(template, mesh, landmark_group=group,
                            data_weights=data_weights(template), verbose=verbose)
    return aligned

def compute_distance(point1, point2):
    diff = point1 - point2
    return np.sqrt(np.sum(diff*diff))

def build_correspondence_matrix(source_mesh, target_mesh, mask, verbose=False):
    if mask is not None:
        source_mesh.landmarks['__lsfm_masked'] = source_mesh.landmarks[
            '__lsfm'].from_mask(mask)
        group = '__lsfm_masked'
        if verbose:
            n_bad_lms = (~mask).sum()
            if not n_bad_lms == 0:
                print('occlusion mask provided with {} False values - '
                      'omitting these landmarks from NICP '
                      'calculation'.format(n_bad_lms))
    else:
        group = '__lsfm'
    #import pdb; pdb.set_trace();
    aligned_mesh, U, tri_indices = non_rigid_icp(source_mesh, target_mesh, landmark_group=group, verbose=verbose)
    n = source_mesh.points.shape[0]
    m = target_mesh.points.shape[0]
    correspondence_matrix = csr_matrix((n,m), dtype=np.bool)
    for idx, tri_index in enumerate(tri_indices):
        target_point_indices = None
        try:
            target_point_indices = target_mesh.trilist[tri_index]
        except:
            pass
        if target_point_indices is None: continue
        distances = np.array([compute_distance(source_mesh.points[idx], target_mesh.points[p_idx]) for p_idx in target_point_indices])
        nearest_point_idx = target_point_indices[np.argmin(distances)]
        correspondence_matrix[idx, nearest_point_idx] = True
    return correspondence_matrix
