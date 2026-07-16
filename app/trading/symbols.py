"""Contract identity across the two IBKR symbol formats.

A tactical Position keeps strike/expiry/right in columns; a statement symbol
encodes them as "TICKER DDMMMYY STRIKE RIGHT". Both reduce to the same
ContractKey so a live position can be paired with its campaign's history.
"""

from __future__ import annotations

from datetime import date, datetime

from app.trading.domain import Position

# (root ticker, expiry, strike, right); the last three are None for stocks.
ContractKey = tuple[str, date | None, float | None, str | None]


def parse_statement_option_symbol(symbol: str) -> tuple[str, date, float, str]:
    root, ddmmmyy, strike, right = symbol.split()
    expiry = datetime.strptime(ddmmmyy, "%d%b%y").date()
    return root, expiry, float(strike), right


def statement_symbol_contract_key(symbol: str) -> ContractKey:
    parts = symbol.split()
    if len(parts) == 4:  # option
        root, expiry, strike, right = parse_statement_option_symbol(symbol)
        return (root, expiry, strike, right)
    return (symbol, None, None, None)  # stock


def position_contract_key(position: Position) -> ContractKey:
    if position.asset_class == "OPT":
        return (position.symbol, position.expiry, position.strike, position.right)
    return (position.symbol, None, None, None)


# IBKR `Listing Exch` code -> Yahoo suffix (verified via yfinance: SHOP.TO,
# SAP.DE, ASML.AS, VOLV-B.ST). US exchanges need no suffix (currency == USD).
_YAHOO_EXCHANGE_SUFFIX = {"TSE": ".TO", "IBIS": ".DE", "AEB": ".AS", "SFB": ".ST"}


def to_yahoo_symbol(position: Position, listing_exch: str | None = None) -> str:
    """The symbol string yfinance expects for this position's live quote.

    Options become an OCC contract symbol (root + YYMMDD + C/P + strike*1000
    zero-padded to 8), matching yfinance's `contractSymbol`. US stocks map to
    the Yahoo ticker (share-class dot -> dash). A non-US stock needs its IBKR
    `listing_exch` (from the statement) to pick the Yahoo suffix; without a
    known one we refuse rather than emit a US-style guess that silently quotes
    the wrong instrument.
    """
    if position.asset_class == "OPT":
        yymmdd = position.expiry.strftime("%y%m%d")
        strike8 = f"{round(position.strike * 1000):08d}"
        return f"{position.symbol}{yymmdd}{position.right}{strike8}"

    ticker = position.symbol.replace(".", "-")
    if position.currency == "USD":
        return ticker
    suffix = _YAHOO_EXCHANGE_SUFFIX.get(listing_exch)
    if suffix is None:
        raise ValueError(
            f"cannot map non-US stock {position.symbol!r} ({position.currency}) "
            f"to a Yahoo symbol without a known listing exchange (got {listing_exch!r})"
        )
    return f"{ticker}{suffix}"
