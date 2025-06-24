from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.factory import make_policy
import numpy as np
import torch

device = "cuda"  # or "cpu"

policy = ACTPolicy.from_pretrained("/mnt/mxy/lerobot/outputs/train/act_franka_libero_0612/checkpoints/060000/pretrained_model")
policy.reset()

# np array of shape (224, 224, 3) for the agent view image
img224_agentview = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
# np array of shape (224, 224, 3) for the eye-in-hand image
img224_eye_in_hand = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
# np array of shape (9) for the joint state
joint_state = np.random.rand(9).astype(np.float32)  # Example joint state

# # Convert images to float32 and normalize to [0, 1]
# img224_agentview = img224_agentview.astype(np.float32) / 255.0
# img224_eye_in_hand = img224_eye_in_hand.astype(np.float32) / 255.0
# Ensure joint state is a numpy array of float32

img224_agentview = torch.from_numpy(img224_agentview) # torch.tensor(img224_agentview, dtype=torch.float32)
img224_eye_in_hand = torch.from_numpy(img224_eye_in_hand) # torch.tensor(img224_eye_in_hand, dtype=torch.float32)
joint_state = torch.from_numpy(joint_state) # torch.tensor(joint_state, dtype=torch.float32)

img224_agentview = img224_agentview.permute(2, 0, 1)
img224_eye_in_hand = img224_eye_in_hand.permute(2, 0, 1)

img224_agentview = img224_agentview.unsqueeze(0)
img224_eye_in_hand = img224_eye_in_hand.unsqueeze(0)
joint_state = joint_state.unsqueeze(0)

# Send data tensors from CPU to GPU
img224_agentview = img224_agentview.to(device, non_blocking=True)
img224_eye_in_hand = img224_eye_in_hand.to(device, non_blocking=True)
joint_state = joint_state.to(device, non_blocking=True)

# Prepare the observation dictionary
obs = {
    "observation.images.image": img224_agentview,
    "observation.images.wrist_image": img224_eye_in_hand,
    "observation.state": joint_state,
    "annotation.human.task_description": f"hahaha",
}
action = policy.select_action(obs)
action = action.squeeze(0)
action = action.to("cpu")