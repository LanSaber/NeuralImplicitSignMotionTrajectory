import pickle
import numpy as np
import os
import math
import warnings
from bisect import bisect_left, bisect_right

keys = ['smplx_root_pose', 
        'smplx_body_pose', 
        'smplx_lhand_pose', 
        'smplx_rhand_pose', 
        'smplx_jaw_pose', 
        'smplx_shape', 
        'smplx_expr'
    ]


def _warn_skip(name, reason):
    warnings.warn(f"Skipping sample {name}: {reason}", RuntimeWarning, stacklevel=2)


def _safe_listdir(path, name, kind="pose directory"):
    if not os.path.isdir(path):
        _warn_skip(name, f"missing {kind}: {path}")
        return None
    try:
        return sorted(os.listdir(path))
    except OSError as exc:
        _warn_skip(name, f"cannot read {kind} {path}: {exc}")
        return None


def _load_pose_sequence(frame_list, name):
    clip_poses = np.zeros([len(frame_list), 179])
    for frame_id, frame in enumerate(frame_list):
        if not os.path.isfile(frame):
            _warn_skip(name, f"missing pose file: {frame}")
            return None
        try:
            with open(frame, 'rb') as f:
                poses = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError) as exc:
            _warn_skip(name, f"cannot load pose file {frame}: {exc}")
            return None

        missing_keys = [key for key in keys if key not in poses]
        if missing_keys:
            _warn_skip(name, f"pose file {frame} is missing keys: {missing_keys}")
            return None

        try:
            pose = np.concatenate([poses[key] for key in keys], 0)
            clip_poses[frame_id] = pose
        except (TypeError, ValueError) as exc:
            _warn_skip(name, f"invalid pose values in {frame}: {exc}")
            return None

    # remove lower body joints and shape, keeping expression
    clip_poses = clip_poses[:, (3 + 3 * 11):]
    clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    return clip_poses


def _load_code(candidates, name):
    for fname in candidates:
        if fname and os.path.isfile(fname):
            return np.load(fname)[0]
    _warn_skip(name, "missing motion code file; tried " + ", ".join(candidates))
    return None


def load_h2s_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    name = ann['name']
    if 'split' in ann:
        split = ann['split']
        base_dir = os.path.join(data_dir, split, 'poses', name)
    else:
        base_dir = os.path.join(data_dir, name)
    clip_text = ann['text']  #csv[csv['SENTENCE_NAME']==basename]['SENTENCE'].item()
    clip_poses = None

    if need_pose:
        fps = float(ann['fps'])
        frame_files = _safe_listdir(base_dir, name)
        if frame_files is None:
            return None, None, None, None
        frame_list = [os.path.join(base_dir, name+'_'+str(frame_id)+'_3D.pkl') for frame_id in range(len(frame_files))]
        if fps > 24:
            target_count = int(24 * len(frame_list) / fps)
            if target_count <= 0:
                _warn_skip(name, f"not enough sampled pose frames from {base_dir}: {target_count}")
                return None, None, None, None
            frame_list = sample(frame_list, count=target_count)
        if len(frame_list) < 4:
            _warn_skip(name, f"not enough pose frames in {base_dir}: {len(frame_list)}")
            return None, None, None, None
        # frame_list = frame_list[:30]
        clip_poses = _load_pose_sequence(frame_list, name)
        if clip_poses is None:
            return None, None, None, None

        #smplx_root_pose (3,)     # 1  Joint
        #smplx_body_pose (63,)    # 21 Joints
        #smplx_lhand_pose (45,)   # 15 Joints
        #smplx_rhand_pose (45,)   # 15 Joints
        #smplx_jaw_pose (3,)      # 1  Joint
        #smplx_shape (10,)        
        #smplx_expr (10,)
        # clip_poses[:,:111] = 0
        # mean = np.mean(clip_poses, axis=0)
        # std = np.std(clip_poses, axis=0)

        # TODO: Completely detele those poses 
        # clip_poses[:, 3: (3 +3*12)] = 0. 
        # clip_poses = np.concatenate((clip_poses[:,:3], clip_poses[:,(3+3*12):]), axis=1)
    
    code = None
    if need_code:
        if not code_path:
            _warn_skip(name, "code_path is not set")
            return None, clip_text, name, None
        code = _load_code([
            os.path.join(code_path, 'how2sign', f'{name}.npy'),
            os.path.join(code_path, f'{name}.npy')
        ], name)

    return clip_poses, clip_text, name, code


def load_csl_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    clip_text = ann['text']
    name = ann['name']
    clip_poses = None
    if need_pose:
        pose_dir = os.path.join(data_dir, 'poses', name)
        frame_files = _safe_listdir(pose_dir, name)
        if frame_files is None:
            return None, None, None, None
        if len(frame_files) < 4:
            _warn_skip(name, f"not enough pose frames in {pose_dir}: {len(frame_files)}")
            return None, None, None, None
        frame_list = [os.path.join(pose_dir, frame) for frame in frame_files]
        clip_poses = _load_pose_sequence(frame_list, name)
        if clip_poses is None:
            return None, None, None, None

    code = None
    if need_code:
        if not code_path:
            _warn_skip(name, "code_path is not set")
            return None, clip_text, name, None
        code = _load_code([
            os.path.join(code_path, 'csl', f'{name}.npy'),
            os.path.join(code_path, f'{name}.npy')
        ], name)

    return clip_poses, clip_text, name, code


def load_iso_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False, dataset=None):
    clip_text = ann['label']
    name = ann['name']
    start, end = ann['start'], ann['end']
    video_file = ann['video_file']
    if dataset in ['csl_iso', 'how2sign_iso']:
        pose_dir = os.path.join(data_dir, 'poses', video_file)
        frame_list = _safe_listdir(pose_dir, name)
        if frame_list is None:
            return None, None, None, None
        frame_idx = [int(x.split('.pkl')[0]) for x in frame_list]
    elif dataset == 'phoenix_iso':
        pose_dir = os.path.join(data_dir, video_file)
        frame_list = _safe_listdir(pose_dir, name)
        if frame_list is None:
            return None, None, None, None
        frame_idx = [int(x.split('.pkl')[0].replace('images', '')) for x in frame_list]
    else:
        _warn_skip(name, f"unsupported isolated dataset: {dataset}")
        return None, None, None, None
    if len(frame_list) < 4:
        _warn_skip(name, f"not enough pose frames in {pose_dir}: {len(frame_list)}")
        return None, None, None, None
    
    start_idx = bisect_left(frame_idx, start)
    end_idx = bisect_right(frame_idx, end)
    frame_list = frame_list[start_idx:end_idx]
    ratio = len(frame_list) / (end-start)
    if ratio < 0.5:
        _warn_skip(name, f"too few aligned pose frames for interval [{start}, {end}]: ratio {ratio:.3f}")
        return None, None, None, None

    clip_poses = None
    if need_pose:
        frame_list = [os.path.join(pose_dir, frame) for frame in frame_list]
        clip_poses = _load_pose_sequence(frame_list, name)
        if clip_poses is None:
            return None, None, None, None

    code = None
    if need_code:
        if not code_path:
            _warn_skip(name, "code_path is not set")
            return None, clip_text, name, None
        if dataset == 'csl_iso':
            src = 'csl'
        elif dataset == 'phoenix_iso':
            src = 'phoenix'
        elif dataset == 'how2sign_iso':
            src = 'how2sign'
        code = _load_code([
            os.path.join(code_path, src, f'{name}.npy'),
            os.path.join(code_path, f'{name}.npy')
        ], name)

    return clip_poses, clip_text, name, code


def load_phoenix_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    clip_text = ann['text']
    name = ann['name']
    clip_poses = None
    if need_pose:
        pose_dir = os.path.join(data_dir, name)
        frame_files = _safe_listdir(pose_dir, name)
        if frame_files is None:
            return None, None, None, None
        if len(frame_files) < 4:
            _warn_skip(name, f"not enough pose frames in {pose_dir}: {len(frame_files)}")
            return None, None, None, None
        frame_list = [os.path.join(pose_dir, frame) for frame in frame_files]
        clip_poses = _load_pose_sequence(frame_list, name)
        if clip_poses is None:
            return None, None, None, None

    code = None
    if need_code:
        if not code_path:
            _warn_skip(name, "code_path is not set")
            return None, clip_text, name, None
        code = _load_code([
            os.path.join(code_path, 'phoenix', f'{name}.npy'),
            os.path.join(code_path, f'{name}.npy')
        ], name)

    return clip_poses, clip_text, name, code


def sample(input,count):
    ss=float(len(input))/count
    return [ input[int(math.floor(i*ss))] for i in range(count) ]


