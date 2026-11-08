"""Synthetic prices for datasets that don't publish them (e.g. WANDS).

Prices are NOT real. They exist so price filters can be demonstrated, and every
record produced here is flagged ``price_is_synthetic=True``.

Each price is log-normal around a per-category median. The random draw is derived
from a hash of (seed, product_id), so a product always gets the same price
regardless of file order or which subset is loaded.
"""
import hashlib
import math
from statistics import NormalDist

PRICE_SEED = 20261101
PRICE_SIGMA = 0.6  # spread of log-price within a category (~±80% at 1σ)
DEFAULT_MEDIAN = 80.0

# Rough medians (USD) per WANDS top-level category; anything else uses DEFAULT_MEDIAN.
CATEGORY_MEDIANS: dict[str, float] = {
    "Furniture": 300.0,
    "Outdoor": 250.0,
    "Rugs": 150.0,
    "Lighting": 120.0,
    "Home Improvement": 90.0,
    "Bed & Bath": 70.0,
    "Storage & Organization": 60.0,
    "Baby & Kids": 80.0,
    "Kitchen & Tabletop": 40.0,
    "Décor & Pillows": 45.0,
}

_NORMAL = NormalDist()


def _uniform(product_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{product_id}".encode()).digest()
    # 53 bits → float in (0, 1); offset keeps it away from exactly 0 or 1.
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53 * (1 - 2e-9) + 1e-9


def synthetic_price(product_id: str, category: str | None, seed: int = PRICE_SEED) -> float:
    median = CATEGORY_MEDIANS.get(category or "", DEFAULT_MEDIAN)
    z = _NORMAL.inv_cdf(_uniform(str(product_id), seed))
    price = median * math.exp(PRICE_SIGMA * z)
    return max(math.floor(price) + 0.99, 0.99)  # retail-style ".99" ending
