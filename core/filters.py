import math
import time

class OneEuroFilter:
    """
    Adaptive low-pass filter for zero-jitter interaction.
    - Slower movement = heavier smoothing.
    - Faster movement = lower latency.
    """
    def __init__(self, min_cutoff=0.3, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-6)
        # Filter the derivative to understand speed
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        self.dx_prev = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.t_prev = t

        # Update cutoff based on current speed
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    @property
    def velocity(self):
        return abs(self.dx_prev) if self.dx_prev is not None else 0.0

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
