import numpy as np
import torch

def random_joint_mask(pose, mask_prob=0.1, maskall=False):
    """
    Randomly masks out some joints in a given pose with probability `mask_prob`.
    If maskall=True, the same joint are masked in all frames.

    Args:
    - pose (numpy array): Shape (timeframes, num_joints, 3), representing a 3D pose sequence.
    - mask_prob (float): Probability of masking each joint.
    
    Returns:
    - masked_pose (numpy array): Pose with missing joints (set to NaN).
    """
    masked_pose = pose.clone()

    if maskall:
        if np.random.rand() < mask_prob:
            joint_index = np.random.randint(0, pose.shape[1])
            masked_pose[:, joint_index, :] = torch.nan
    else:
        mask = torch.from_numpy(np.random.rand(*pose.shape[:2]) < mask_prob).to(pose.device)
        masked_pose[mask] = torch.nan
    return masked_pose

def structured_missing_joints(pose, mask_prob=0.1, maskall=False):
    """
    For each frame, randomly selects and masks all joints of one limb with a given probability.
    
    Args:
    - pose (numpy array): Shape (timeframes, num_joints, 3), representing a 3D motion sequence.
    - mask_prob (float): Probability of masking a randomly chosen limb in each frame.
    
    Returns:
    - masked_pose (numpy array): Pose with missing limbs.
    - chosen_limbs (list): List of selected limbs for each frame.
    """
    limb_indices = {
        "left_lower_arm": [16, 17],  # Example indices for left elbow, wrist
        "right_lower_arm": [21, 22],
        "left_leg": [6, 7, 8],
        "right_leg": [1, 2, 3],
        "head": [13, 14],
        "left_hand": [18, 19],
        "right_hand": [23, 24],
        "feet": [4, 5, 9, 10]
    }
    
    masked_pose = pose.clone()
    chosen_limbs = []
    num_frames = pose.shape[0]  # Number of time frames

    if maskall:
        # Check if there are no NaN values in the frames
        if not torch.isnan(masked_pose).any():
            chosen_limb = np.random.choice(list(limb_indices.keys()))
            chosen_limbs = [chosen_limb] * num_frames

            if np.random.rand() < mask_prob:
                limb_idx = torch.tensor(limb_indices[chosen_limb], dtype=torch.long, device=pose.device)
                masked_pose[:, limb_idx, :] = torch.nan

    for i in range(num_frames):
        # Check if there are no NaN values in the current frame
        if not torch.isnan(masked_pose[i]).any():
            # Randomly choose a limb for this frame
            chosen_limb = np.random.choice(list(limb_indices.keys()))
            chosen_limbs.append(chosen_limb)

            # Apply probability check for masking
            if np.random.rand() < mask_prob:
                limb_idx = torch.tensor(limb_indices[chosen_limb], dtype=torch.long, device=pose.device)
                masked_pose[i, limb_idx, :] = torch.nan  # or set to zero

    return masked_pose, chosen_limbs

def temporal_occlusion(sequence, missing_frame_prob=0.05, mask_prob=1):
    """
    Randomly masks 3–10 consecutive frames, applying the same joint/limb mask
    across all frames in that block (i.e., occlusion window).
    
    Args:
    - sequence (torch.Tensor): Shape (T, num_joints, 3)
    - missing_frame_prob (float): Probability that a block of frames will be masked
    - mask_prob (float): Probability for masking joints/limbs inside each frame

    Returns:
    - masked_sequence (torch.Tensor): Sequence with masked blocks
    """
    num_frames = sequence.shape[0]
    masked_sequence = sequence.clone()
    i = 0

    while i < num_frames:
        if np.random.rand() < missing_frame_prob:
            num_to_mask = np.random.randint(3, 11)
            end_idx = min(i + num_to_mask, num_frames)

            window = sequence[i:end_idx]
            if not torch.isnan(window).any():
                if np.random.rand() < 0.5:
                    masked_window = random_joint_mask(window, mask_prob=mask_prob, maskall=True)
                else:
                    masked_window, _ = structured_missing_joints(window, mask_prob=mask_prob, maskall=True)
                print(f"Temporal Making {num_to_mask} Frames")
                masked_sequence[i:end_idx] = masked_window
            i = end_idx
        else:
            i += 1

    return masked_sequence



