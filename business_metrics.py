"""
RTIE — Business Impact Metrics

Maps fault classifications to real-world business KPIs:
    - Estimated energy loss percentage
    - Annual cost impact (£/year)
    - CO₂ reduction potential (kg CO₂/year)
"""

import config


def compute_business_impact(fault_label: str, efficiency_score: float = None) -> dict:
    """
    Compute business impact metrics for a given fault classification.

    Args:
        fault_label: One of the 5 fault classes.
        efficiency_score: Optional model-predicted efficiency (0-100).

    Returns:
        dict with energy_loss_pct, estimated_annual_cost_gbp, co2_reduction_potential_kg
    """
    # Energy loss from lookup
    energy_loss_pct = config.ENERGY_LOSS_MAP.get(fault_label, 0.0)

    # Annual energy consumption (kWh)
    annual_kwh = (
        config.RADIATOR_POWER_KW
        * config.DAILY_HOURS
        * config.HEATING_DAYS_PER_YEAR
    )

    # Wasted energy due to fault
    wasted_kwh = annual_kwh * (energy_loss_pct / 100.0)

    # Cost impact
    estimated_annual_cost_gbp = round(wasted_kwh * config.ENERGY_PRICE_GBP_PER_KWH, 2)

    # CO₂ impact
    co2_reduction_potential_kg = round(wasted_kwh * config.CO2_KG_PER_KWH, 2)

    return {
        "energy_loss_pct": energy_loss_pct,
        "wasted_kwh_per_year": round(wasted_kwh, 1),
        "estimated_annual_cost_gbp": estimated_annual_cost_gbp,
        "co2_reduction_potential_kg": co2_reduction_potential_kg,
    }


def generate_impact_summary():
    """Generate business impact summary for all classes."""
    import pandas as pd
    import os

    rows = []
    for fault in config.CLASS_NAMES:
        impact = compute_business_impact(fault)
        rows.append({"fault_class": fault, **impact})

    df = pd.DataFrame(rows)

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    csv_path = os.path.join(config.REPORT_DIR, "business_impact_summary.csv")
    df.to_csv(csv_path, index=False)

    print("Business Impact Summary")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved to {csv_path}")

    return df


if __name__ == "__main__":
    generate_impact_summary()
