"""Parser for the IBKR tactical book export (flat Flex table).

Turns the fully-quoted 12-column CSV into typed `Position` records. Carries
only trustworthy raw facts: `CostBasisPrice` (always 0 in this export) and
`PercentOfNAV` (wrong denominator) are dropped.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

from app.trading.domain import Position


def parse_tactical_book(content: str) -> list[Position]:
    reader = csv.DictReader(io.StringIO(content.removeprefix("\ufeff")))
    positions: list[Position] = []
    for row in reader:
        is_option = row["AssetClass"] == "OPT"
        positions.append(
            Position(
                # Option Symbol is a padded OCC string ("FAKE  260717C\u2026");
                # the root ticker is its first whitespace-delimited token.
                symbol=row["Symbol"].split()[0],
                asset_class=row["AssetClass"],
                currency=row["CurrencyPrimary"],
                fx_rate_to_base=float(row["FXRateToBase"]),
                quantity=float(row["Quantity"]),
                mark_price=float(row["MarkPrice"]),
                position_value=float(row["PositionValue"]),
                strike=float(row["Strike"]) if is_option else None,
                expiry=(
                    datetime.strptime(row["Expiry"], "%d/%m/%Y").date()
                    if is_option
                    else None
                ),
                right=row["Put/Call"] if is_option else None,
            )
        )
    return positions
