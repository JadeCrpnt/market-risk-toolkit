"""Pricing d'options européennes et américaines : Black-Scholes, Greeks,
volatilité implicite et arbre binomial de Cox-Ross-Rubinstein."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def payoff_call(ST, K):
    """Payoff d'un call à maturité : max(ST - K, 0)."""
    return np.maximum(ST - K, 0)


def payoff_put(ST, K):
    """Payoff d'un put à maturité : max(K - ST, 0)."""
    return np.maximum(K - ST, 0)


def _d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    """
    Prix Black-Scholes d'une option européenne.

    S : spot, K : strike, T : maturité en années, r : taux sans risque,
    sigma : volatilité, option : "call" ou "put", q : rendement du dividende.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)




def bs_greeks(S, K, T, r, sigma, option="call", q=0.0):
    """
    Greeks Black-Scholes en conventions de marché :
    vega et rho pour 1 point, theta par jour calendaire.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    Sq = S * np.exp(-q * T)
    Kr = K * np.exp(-r * T)

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = Sq * norm.pdf(d1) * np.sqrt(T) / 100
    decay = -Sq * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

    if option == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (decay + q * Sq * norm.cdf(d1) - r * Kr * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (decay - q * Sq * norm.cdf(-d1) + r * Kr * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, option="call", q=0.0, low=1e-4, high=5.0):
    """Volatilité implicite par la méthode de Brent. Renvoie NaN sans solution."""
    f = lambda s: bs_price(S, K, T, r, s, option, q) - price
    try:
        return brentq(f, low, high)
    except ValueError:
        return np.nan


def binomial_price(S, K, T, r, sigma, n=500, option="call", american=False, q=0.0):
    """Prix d'une option européenne ou américaine par arbre binomial à n pas."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    payoff = payoff_call if option == "call" else payoff_put

    j = np.arange(n + 1)
    V = payoff(S * u**j * d**(n - j), K)

    for i in range(n - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if american:
            k = np.arange(i + 1)
            V = np.maximum(V, payoff(S * u**k * d**(i - k), K))

    return V[0]