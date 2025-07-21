import numpy as np

class KalmanFilterEstimator:
    """
    Online Kalman filter to estimate the time-varying intercept (alpha_t)
    and hedge ratio (beta_t) in the linear relationship:
        p1_t = alpha_t + beta_t * p2_t + epsilon_t
    """

    def __init__(self,
                 delta: float = 1e-5,
                 obs_var: float = 1e-3):
        """
        Args:
            delta     – Process noise parameter; larger means more weight on new data.
            obs_var   – Observation noise variance (measurement uncertainty).
        """
        # State vector: [alpha, beta]
        self.state = np.zeros(2, dtype=float)

        # State covariance matrix
        self.P = np.eye(2, dtype=float)

        # Process (model) noise covariance
        # R = (delta / (1 - delta)) * I
        self.R = np.eye(2, dtype=float) * (delta / (1.0 - delta))

        # Observation noise variance
        self.obs_var = obs_var

    def update(self, p1: float, p2: float):
        """
        Incorporate a new observation (p1, p2) to update alpha, beta, and compute the spread.

        Args:
            p1 – Price of asset 1 at time t
            p2 – Price of asset 2 at time t

        Returns:
            alpha   – Updated intercept estimate
            beta    – Updated hedge ratio estimate
            spread  – p1 - (alpha + beta * p2)
        """
        # 1) Predict step: add process noise to covariance
        self.P = self.P + self.R

        # 2) Observation design matrix H = [1, p2]
        H = np.array([1.0, p2])

        # 3) Innovation covariance S = H P Hᵀ + obs_var
        S = H @ self.P @ H.T + self.obs_var

        # 4) Kalman gain K = P Hᵀ / S
        K = (self.P @ H.T) / S

        # 5) Measurement residual (innovation)
        y_pred = H @ self.state
        residual = p1 - y_pred

        # 6) State update
        self.state = self.state + K * residual

        # 7) Covariance update
        self.P = self.P - np.outer(K, H @ self.P)

        alpha, beta = self.state
        spread = p1 - (alpha + beta * p2)

        return alpha, beta, spread

    def reset(self,
              alpha0: float = 0.0,
              beta0: float = 0.0,
              P0: np.ndarray = None):
        """
        Re-initialize filter state and covariance.

        Args:
            alpha0 – Initial intercept
            beta0  – Initial hedge ratio
            P0     – Optional 2×2 initial covariance matrix
        """
        self.state = np.array([alpha0, beta0], dtype=float)
        self.P = P0.copy() if P0 is not None else np.eye(2, dtype=float)
