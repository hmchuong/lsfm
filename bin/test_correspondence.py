import os
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm

from scipy.sparse import save_npz
import lsfm.io as lio
from lsfm import correspondence_meshes
import warnings
warnings.filterwarnings('ignore')

#target = lio.import_mesh(Path("/home/chuonghm/Projects/lsfm/input/F0001_SU04WH_F3D.obj"))
#source_texture = lio.import_mesh(Path("/home/chuonghm/Projects/lsfm/input/F0001_AN01WH_F3D.obj"))
#source = lio.import_mesh_as_color_trimesh(Path("/home/chuonghm/Projects/lsfm/input/F0001_AN01WH_F3D.obj"))

# mat = correspondence_meshes((source_texture, source), target, True)

def correspondence(target_path):
    #print(target_path)
    
    target = lio.import_mesh(Path(target_path))
    try:
        mat = correspondence_meshes("template", target, False)
        save_npz(target_path.replace(".obj", "_correspondence.npz"), mat)
    except:
        print("Cannot correspond", target_path)
        pass

root_dir = "/media/chuonghm/data/BU_3DFE/correspondence"

params = []
for entry in os.scandir(root_dir):
    if entry.name.startswith(".") and entry.is_dir(): continue
    if not entry.is_dir(): continue
    subdir = os.path.join(root_dir, entry.name)
    for sub_entry in os.scandir(subdir):
        if sub_entry.is_dir() or not sub_entry.name.endswith(".obj"): continue
        target_path = os.path.join(subdir, sub_entry.name)
        if os.path.exists(target_path.replace(".obj", "_correspondence.npz")): continue
        params.append(target_path)

with Pool(processes=4) as p:
        max_ = len(params)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(correspondence, params))):
                pbar.update()
