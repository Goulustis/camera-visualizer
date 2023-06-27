import json
import os.path as osp
import glob
import numpy as np


def load_camera_json(json_f):

    with open(json_f, "r") as f:
        data = json.load(f)
        return data["orientation"], data["position"]

def load_cameras(cam_dir):

    cam_fs = sorted(glob.glob(osp.join(cam_dir, "*.json")))
    
    rots, trans = [], []

    for cam_f in cam_fs:
        rot, tran = load_camera_json(cam_f)
        rots.append(rot), trans.append(tran)

    return np.stack(rots), np.stack(trans)
