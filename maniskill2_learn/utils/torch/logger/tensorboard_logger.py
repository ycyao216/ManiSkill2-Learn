import os.path as osp
import numbers, numpy as np
import matplotlib.pyplot as plt

from .base_logger import BaseLogger

class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir=None):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(osp.join(log_dir, "tf_logs"))

    def log(self, tags, n_iter, tag_name="train"):
        tags = self.get_loggable_tags(tags, tag_name=tag_name)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, n_iter)
            elif np.isscalar(val) or val.size == 1:
                self.writer.add_scalar(tag, val, n_iter)
            else:
                if val.ndim == 2:
                    cmap = plt.get_cmap('jet')
                    val = cmap(val)[..., :3]
                assert val.ndim == 3, f"Image should have two dimension! You provide: {tag, val.shape}!"
                self.writer.add_image(tag, val, n_iter, dataformats='HWC')

    def close(self):
        self.writer.close()
