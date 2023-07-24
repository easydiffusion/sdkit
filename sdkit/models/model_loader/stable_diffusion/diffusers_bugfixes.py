from diffusers.schedulers import DPMSolverSinglestepScheduler
import numpy as np
import torch

# temporary patch, while waiting for PR: https://github.com/huggingface/diffusers/pull/4231

old_set_timesteps = DPMSolverSinglestepScheduler.set_timesteps


def set_timesteps_remove_duplicates(self, num_inference_steps: int, device=None):
    old_set_timesteps(self, num_inference_steps, device)

    timesteps = self.timesteps.cpu().detach().numpy().astype(np.int64)

    # when num_inference_steps == num_train_timesteps, we can end up with
    # duplicates in timesteps.
    _, unique_indices = np.unique(timesteps, return_index=True)
    timesteps = timesteps[np.sort(unique_indices)]

    self.timesteps = torch.from_numpy(timesteps).to(device)

    self.num_inference_steps = len(timesteps)

    self.order_list = self.get_order_list(self.num_inference_steps)


DPMSolverSinglestepScheduler.set_timesteps = set_timesteps_remove_duplicates
