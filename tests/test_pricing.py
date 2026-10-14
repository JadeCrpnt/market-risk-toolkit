import numpy as np
from src.pricing import (payoff_call, payoff_put, bs_price, bs_greeks,
                         implied_vol, binomial_price)

S, K, T, r, SIGMA = 100, 100, 1, 0.05, 0.20


def test_payoffs():
    assert payoff_call(120, 100) == 20
    assert payoff_call(80, 100) == 0
    assert payoff_put(80, 100) == 20
    assert payoff_put(120, 100) == 0


def test_bs_valeurs_de_reference():
    assert abs(bs_price(S, K, T, r, SIGMA, "call") - 10.4506) < 1e-3
    assert abs(bs_price(S, K, T, r, SIGMA, "put") - 5.5735) < 1e-3


def test_parite_call_put_avec_dividende():
    for s, k, t, rr, v, q in [(100, 100, 1, 0.05, 0.2, 0.0),
                              (80, 110, 0.5, 0.02, 0.35, 0.01),
                              (150, 90, 2, 0.01, 0.15, 0.03)]:
        c = bs_price(s, k, t, rr, v, "call", q)
        p = bs_price(s, k, t, rr, v, "put", q)
        assert abs((c - p) - (s * np.exp(-q * t) - k * np.exp(-rr * t))) < 1e-10


def test_delta_et_gamma_par_differences_finies():
    h = 0.01
    up = bs_price(S + h, K, T, r, SIGMA)
    mid = bs_price(S, K, T, r, SIGMA)
    down = bs_price(S - h, K, T, r, SIGMA)
    g = bs_greeks(S, K, T, r, SIGMA)
    assert abs(g["delta"] - (up - down) / (2 * h)) < 1e-6
    assert abs(g["gamma"] - (up - 2 * mid + down) / h**2) < 1e-4


def test_vol_implicite_retrouve_la_vol():
    prix = bs_price(S, K, T, r, 0.25, "put", q=0.01)
    assert abs(implied_vol(prix, S, K, T, r, "put", q=0.01) - 0.25) < 1e-6


def test_vol_implicite_sans_solution():
    assert np.isnan(implied_vol(0.0001, S, 50, T, r, "call"))


def test_binomial_converge_vers_bs():
    assert abs(binomial_price(S, K, T, r, SIGMA, n=1000) - bs_price(S, K, T, r, SIGMA)) < 0.01


def test_call_americain_egal_europeen_sans_dividende():
    euro = binomial_price(S, K, T, r, SIGMA, n=300, option="call")
    amer = binomial_price(S, K, T, r, SIGMA, n=300, option="call", american=True)
    assert abs(amer - euro) < 1e-8


def test_put_americain_plus_cher():
    euro = binomial_price(S, K, T, r, SIGMA, n=300, option="put")
    amer = binomial_price(S, K, T, r, SIGMA, n=300, option="put", american=True)
    assert amer > euro