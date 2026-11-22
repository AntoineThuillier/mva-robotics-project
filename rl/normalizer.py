"""Running mean and standard deviation (replaces MimicKit's learning/normalizer.py).

    norm = Normalizer(dim)
    norm.record(x)        # accumulate a batch (any leading shape, last dims = dim)
    norm.update()         # merge the accumulated batches into the running mean and std
    norm.normalize(x)     # (x - mean) / std, clipped to +-clip

Same math as MimicKit: the running mean of x and of x^2 are merged with weights proportional
to the sample counts; var = mean(x^2) - mean^2, floored at min_std^2.
"""

import torch


class Normalizer(torch.nn.Module):

    def __init__(self, dim, device, clip=10.0, min_std=1e-4, init_mean=None, init_std=None):
        super().__init__()
        shape = (dim,) if isinstance(dim, int) else tuple(dim)
        self._clip = clip
        self._min_var = min_std ** 2
        # buffers: saved with the model, moved with .to(), not trained
        self.register_buffer("count", torch.zeros(1, dtype=torch.long, device=device))
        self.register_buffer("mean", torch.zeros(shape, device=device) if init_mean is None
                             else torch.as_tensor(init_mean, dtype=torch.float32, device=device).clone())
        self.register_buffer("std", torch.ones(shape, device=device) if init_std is None
                             else torch.as_tensor(init_std, dtype=torch.float32, device=device).clone())
        self.register_buffer("mean_sq", self.std ** 2 + self.mean ** 2)
        self._new_count = 0
        self._new_sum = torch.zeros(shape, device=device)
        self._new_sum_sq = torch.zeros(shape, device=device)

    def record(self, x):
        x = x.reshape(-1, *self.mean.shape)
        self._new_count += x.shape[0]
        self._new_sum += x.sum(0)
        self._new_sum_sq += (x ** 2).sum(0)

    def update(self):
        if self._new_count == 0:
            return
        new_total = self.count + self._new_count
        w_old = self.count.float() / new_total.float()
        w_new = float(self._new_count) / new_total.float()
        self.mean[:] = w_old * self.mean + w_new * self._new_sum / self._new_count
        self.mean_sq[:] = w_old * self.mean_sq + w_new * self._new_sum_sq / self._new_count
        self.count[:] = new_total
        self.std[:] = torch.sqrt(torch.clamp_min(self.mean_sq - self.mean ** 2, self._min_var))
        self._new_count = 0
        self._new_sum.zero_()
        self._new_sum_sq.zero_()

    def normalize(self, x):
        return torch.clamp((x - self.mean) / self.std, -self._clip, self._clip)

    def unnormalize(self, norm_x):
        return norm_x * self.std + self.mean
