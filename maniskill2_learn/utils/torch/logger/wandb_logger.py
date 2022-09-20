from .base_logger import BaseLogger
import wandb
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


class WandbLogger(BaseLogger):
    def __init__(self, wandb_key:str, log_dir: str, wandb_cfg: dict):
        wandb.login(key=wandb_key)
        wandb.init(
            dir = log_dir,
            **wandb_cfg
        )

    def log(self, tags, n_iter, tag_name="train"):
        ret_dict = dict()
        tags = self.get_loggable_tags(tags, tag_name=tag_name)
        for tag, val in tags.items():
            if np.isscalar(val) or val.size == 1 or isinstance(val, str):
                ret_dict[tag] = val 
            elif isinstance(val,np.ndarray) and val.ndim==1:
                ret_dict[tag] = np.linalg.norm(val)
            else:
                if val.ndim == 2:
                    cmap = plt.get_cmap('jet')
                    val = cmap(val)[..., :3]
                assert val.ndim == 3, f"Image should have two dimension! You provide: {tag, val.shape}!"
                ret_dict[tag] = wandb.Image(val)
        wandb.log(ret_dict, step=n_iter, commit=True)






