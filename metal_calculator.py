"""
Metal recovery value calculator for mixed municipal e-waste.

Spot prices as of February 19, 2026:
  Gold:      $5,006 / troy oz  (source: Fortune / spot market)
  Palladium: $1,752 / troy oz  (source: APMEX spot)
  Copper:    $4.50  / lb scrap payout (~76% of $5.90 spot, after processing)

Yield per ton of mixed consumer e-waste (phones, laptops, TVs, appliances):
  Gold:      0.50 troy oz / ton
  Palladium: 0.05 troy oz / ton
  Copper:    70 lb / ton

Note: Yields assume mixed municipal e-waste, not pure PCB feedstock.
Pure PCB yields are 5–7x higher. Rates should be refreshed quarterly.
"""

# --- Live-pegged spot prices (Feb 19, 2026) ---
GOLD_SPOT_PER_OZ = 5006.00       # USD per troy oz
PALLADIUM_SPOT_PER_OZ = 1752.00  # USD per troy oz
COPPER_SCRAP_PER_LB = 4.50       # USD per lb (scrap payout after processing discount)

# --- Recovery yield per ton of mixed municipal e-waste ---
GOLD_OZ_PER_TON = 0.50      # troy oz
PALLADIUM_OZ_PER_TON = 0.05  # troy oz
COPPER_LB_PER_TON = 70.0    # lbs


def calculate_recovery_value(tonnage: float) -> dict:
    """
    Estimates scrap metal recovery value for a given tonnage of mixed e-waste.

    Args:
        tonnage: Tons of mixed municipal e-waste.

    Returns:
        Dict with per-metal breakdown and total value:
        {
            "gold":      {"yield_units": str, "value": float},
            "palladium": {"yield_units": str, "value": float},
            "copper":    {"yield_units": str, "value": float},
            "total":     float
        }
    """
    gold_oz = GOLD_OZ_PER_TON * tonnage
    gold_value = gold_oz * GOLD_SPOT_PER_OZ

    palladium_oz = PALLADIUM_OZ_PER_TON * tonnage
    palladium_value = palladium_oz * PALLADIUM_SPOT_PER_OZ

    copper_lb = COPPER_LB_PER_TON * tonnage
    copper_value = copper_lb * COPPER_SCRAP_PER_LB

    total = gold_value + palladium_value + copper_value

    return {
        "gold":      {"yield_units": f"{gold_oz:.3f} troy oz", "value": gold_value},
        "palladium": {"yield_units": f"{palladium_oz:.3f} troy oz", "value": palladium_value},
        "copper":    {"yield_units": f"{copper_lb:.1f} lbs", "value": copper_value},
        "total":     total,
    }
