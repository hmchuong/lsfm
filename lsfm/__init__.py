from .landmark import landmark_mesh, get_landmark_points, LANDMARK_MASK
from .visualize import visualize_nicp_result
from .correspond import correspond_mesh, build_correspondence_matrix
from .data.basel import load_basel_template_metadata
from .data import prepare_mesh_as_template, load_template
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def landmark_and_correspond_mesh(mesh, verbose=False):
    mesh = mesh.copy()
    lms = landmark_mesh(mesh, verbose=verbose)
    mesh.landmarks['__lsfm_masked'] = lms['landmarks_3d_masked']
    shape = correspond_mesh(mesh, mask=lms['occlusion_mask'],
                                      verbose=verbose),
    return_dict = {
        'shape_nicp': shape[0],
        'landmarked_image': lms['landmarked_image'],
        'U': shape[1],
        'tri_indices': shape[2]
    }
    return_dict['shape_nicp_visualization'] = visualize_nicp_result(
        return_dict['shape_nicp'])

    return return_dict

def correspondence_meshes(source_mesh, target_mesh, verbose=False):
    
    target_mesh = target_mesh.copy()
    # Detect landmark for source mesh
    if source_mesh != "template":
        texture_mesh, color_mesh = source_mesh
        lmpts = get_landmark_points(texture_mesh)
        
        meta = load_basel_template_metadata()
        ibug68 = meta['landmarks']['ibug68']
        ibug68 = ibug68.from_mask(LANDMARK_MASK)
        ibug68.points = lmpts.points
        nosetip = meta['landmarks']['nosetip']
        nosetip.points = ((2*lmpts.points[30] + 1*lmpts.points[33])/3).reshape(1, -1)
        
        color_mesh.landmarks['ibug68'] = ibug68
        color_mesh.landmarks['nosetip'] = nosetip
        color_mesh = prepare_mesh_as_template(color_mesh)
        source_mesh = color_mesh.copy()
    else:
        source_mesh = load_template().copy()
    
    lms = landmark_mesh(target_mesh, verbose=verbose)
    target_mesh.landmarks['__lsfm_masked'] = lms['landmarks_3d_masked']
    #import pdb; pdb.set_trace()
    mat = build_correspondence_matrix(source_mesh, target_mesh,lms['occlusion_mask'],verbose=verbose)
    return mat
    