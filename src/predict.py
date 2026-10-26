"""
Command-line predictor for Pavement Roughness (IRI) and Condition Assessment.
Usage:
    python src/predict.py --sn 8.5 --aadt 4500
"""

import argparse
from model import predict_piecewise_domain, classify_pavement_condition


def main():
    parser = argparse.ArgumentParser(
        description="Predict Pavement International Roughness Index (IRI) & Maintenance Action."
    )
    parser.add_argument(
        "--sn",
        type=float,
        required=True,
        help="Effective Pavement Structural Number (Total SN) from HWD analysis (typically 4.0 - 15.0).",
    )
    parser.add_argument(
        "--aadt",
        type=float,
        required=True,
        help="Annual Average Daily Traffic (vehicles per day, e.g., 4500).",
    )
    args = parser.parse_args()

    pred_iri_in_mi = predict_piecewise_domain(args.sn, args.aadt)
    pred_iri_m_km = pred_iri_in_mi * 0.0157828
    rating, ride, action = classify_pavement_condition(pred_iri_in_mi)

    print("\n" + "=" * 60)
    print("   PAVEMENT CONDITION & ROUGHNESS (IRI) PREDICTION")
    print("=" * 60)
    print(f" Input Structural Number (SN) : {args.sn:.2f}")
    print(f" Input Traffic Volume (AADT)  : {args.aadt:,.0f} veh/day")
    print("-" * 60)
    print(f" Predicted IRI (in/mi)        : {pred_iri_in_mi:.2f} in/mi")
    print(f" Predicted IRI (m/km)         : {pred_iri_m_km:.3f} m/km")
    print(f" Condition Classification     : {rating}")
    print(f" Ride Quality Standard        : {ride}")
    print(f" Recommended Maintenance      : {action}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
