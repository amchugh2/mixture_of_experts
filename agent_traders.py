import numpy as np

class BaseTrader:
    def __init__(self, trader_id, initial_price, sigma_bounds=(0.15, 0.30), lambda_=0.5, sigma_J=0.02):
        self.trader_id = trader_id
        self.last_price = initial_price
        self.order_id = trader_id * 10000

        self.sigma = np.random.uniform(*sigma_bounds)
        self.lambda_ = lambda_
        self.sigma_J = sigma_J

    def jump_term(self, dt):
        dN = np.random.poisson(self.lambda_ * dt)
        return np.random.normal(0, self.sigma_J) * dN

    def brownian_term(self, dt):
        return self.sigma * np.sqrt(dt) * np.random.normal()

    def apply_quote_band(self, fair_price, delta=0.002):
        """Constrain quote to ±0.2% around fair value."""
        if np.random.rand() < 0.5:
            return fair_price * (1 - delta)
        else:
            return fair_price * (1 + delta)


class MeanRevertingTrader(BaseTrader):
    def __init__(self, trader_id, initial_price, theta_bounds=(2.5, 5.0), K=20, **kwargs):
        super().__init__(trader_id, initial_price, **kwargs)
        self.theta = np.random.uniform(*theta_bounds)
        self.K = K

    def get_order(self, t, history):
        if t < self.K:
            return None
        dt = 1.0
        log_Pt = np.log(history[t - 1])
        log_mean = np.log(np.mean(history[t - self.K:t]))
        drift = self.theta * (log_mean - log_Pt)

        log_fair = log_Pt + drift * dt + self.brownian_term(dt) + self.jump_term(dt)
        fair_price = np.exp(log_fair)
        quoted_price = self.apply_quote_band(fair_price)

        side = 'bid' if drift > 0 else 'ask'
        return {
            'type': 'limit',
            'side': side,
            'quantity': 100,
            'price': round(quoted_price, 2),
            'trade_id': self.order_id + t
        }


class TrendFollowingTrader(BaseTrader):
    def __init__(self, trader_id, initial_price, alpha_bounds=(1.0, 2.0), L=5, **kwargs):
        super().__init__(trader_id, initial_price, **kwargs)
        self.alpha = np.random.uniform(*alpha_bounds)
        self.L = L

    def get_order(self, t, history):
        if t < self.L:
            return None
        dt = 1.0
        log_Pt = np.log(history[t - 1])
        log_P_past = np.log(history[t - self.L])
        drift = self.alpha * (log_Pt - log_P_past)

        log_fair = log_Pt + drift * dt + self.brownian_term(dt) + self.jump_term(dt)
        fair_price = np.exp(log_fair)
        quoted_price = self.apply_quote_band(fair_price)

        side = 'bid' if drift > 0 else 'ask'
        return {
            'type': 'limit',
            'side': side,
            'quantity': 100,
            'price': round(quoted_price, 2),
            'trade_id': self.order_id + t
        }

class NoiseTrader(BaseTrader):
    def get_order(self, t, history):
        if t < 1:
            return None
        dt = 1.0
        log_Pt = np.log(history[t - 1])
        log_fair = log_Pt + self.brownian_term(dt) + self.jump_term(dt)
        fair_price = np.exp(log_fair)
        quoted_price = self.apply_quote_band(fair_price)

        side = np.random.choice(['bid', 'ask'])
        return {
            'type': 'limit',
            'side': side,
            'quantity': 100,
            'price': round(quoted_price, 2),
            'trade_id': self.order_id + t
        }
