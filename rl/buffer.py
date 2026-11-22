"""Rollout buffer (replaces MimicKit's learning/experience_buffer.py).

Holds one iteration of data: steps_per_iter policy steps x num_envs environments, one tensor
per field, shape (T, N, ...). Fields are created on their first record.
"""

import torch


class RolloutBuffer:

    def __init__(self, steps, num_envs, device):
        self.steps, self.num_envs, self.device = steps, num_envs, device
        self.data = {}
        self.t = 0
        self._new_permutation()

    def reset(self):
        """Start of an iteration: back to step 0, new sampling permutation."""
        self.t = 0
        self._new_permutation()

    def _new_permutation(self):
        self._perm = torch.randperm(self.steps * self.num_envs, device=self.device)
        self._head = 0

    def record(self, name, x):
        """Stores x (num_envs, ...) as field `name` at the current step."""
        if name not in self.data:
            self.data[name] = torch.zeros((self.steps, *x.shape), dtype=x.dtype, device=self.device)
        self.data[name][self.t] = x

    def next_step(self):
        self.t += 1

    def flat(self, name):
        """(T * N, ...) view of a field."""
        x = self.data[name]
        return x.view(-1, *x.shape[2:])

    def set(self, name, x):
        """Sets a whole field, x of shape (T, N, ...)."""
        self.data[name] = x

    def minibatches(self, batch_size, num_steps):
        """num_steps minibatches of batch_size samples, drawn without replacement from a random
        permutation of all T * N samples, renewed when used up. As MimicKit, the permutation
        carries over between calls within an iteration (the actor's minibatches continue where
        the critic's stopped) and is renewed by reset()."""
        n = self.steps * self.num_envs
        for _ in range(num_steps):
            if self._head + batch_size <= n:
                idx = self._perm[self._head:self._head + batch_size]
                self._head += batch_size
            else:
                rest = self._perm[self._head:]
                self._new_permutation()
                self._head = batch_size - rest.shape[0]
                idx = torch.cat([rest, self._perm[:self._head]])
            yield {k: self.flat(k)[idx] for k in self.data}
