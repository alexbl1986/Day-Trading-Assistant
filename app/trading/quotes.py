"""Live-quote lookup: a position -> its current (or last-known) market Quote.

The one network-touching line (the yfinance call) lives behind the `fetch`
seam, so all the resolution/guard logic is tested with a fake fetch. get_quote
never hides staleness: the Quote carries the price's own timestamp and the
market state, and downstream tools decide whether that's fresh enough to act on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from app.trading.domain import Position, Quote
from app.trading.symbols import to_yahoo_symbol

# What `fetch` returns: a normalized dict adapted from yfinance `.info`, or None
# when the venue has no quote for the symbol.
RawQuote = dict[str, object]


def _yahoo_fetch(yahoo_symbol: str) -> RawQuote | None:
    """The one network-touching line: yfinance `.info` -> our normalized shape.

    Verified (probe): `.info` carries regularMarketPrice / regularMarketTime
    (epoch secs) / marketState / currency for BOTH stocks and option contract
    symbols. A missing price means no quote (delisted / bad symbol).
    """
    import yfinance as yf

    info = yf.Ticker(yahoo_symbol).info
    price = info.get("regularMarketPrice")
    if price is None:
        return None
    return {
        "price": price,
        "currency": info.get("currency"),
        "epoch": info.get("regularMarketTime"),
        "market_state": info.get("marketState"),
    }


def quote_yahoo_symbol(
    yahoo_symbol: str, fetch: Callable[[str], RawQuote | None] = _yahoo_fetch
) -> Quote:
    """Quote a bare Yahoo symbol (index/ETF/stock: `^VIX`, `SPY`, `QQQ`).

    The market-regime path has no `Position` to cross-check a currency against,
    so this skips that guard but keeps everything else `get_quote` promises:
    fail-loud on no quote, and the price's own timestamp + market state so a
    stale close can never pass for a live tick.
    """
    raw = fetch(yahoo_symbol)
    if raw is None or raw.get("price") is None:
        raise ValueError(f"no quote available for {yahoo_symbol!r} (delisted or bad symbol?)")
    return Quote(
        symbol=yahoo_symbol,
        price=float(raw["price"]),
        currency=str(raw["currency"]),
        as_of=datetime.fromtimestamp(int(raw["epoch"]), tz=timezone.utc),
        market_state=str(raw["market_state"]),
    )


def get_quote(
    position: Position,
    listing_exch: str | None = None,
    fetch: Callable[[str], RawQuote | None] = _yahoo_fetch,
) -> Quote:
    yahoo = to_yahoo_symbol(position, listing_exch)
    raw = fetch(yahoo)
    if raw is None or raw.get("price") is None:
        raise ValueError(f"no quote available for {yahoo!r} (delisted or bad symbol?)")
    currency = str(raw["currency"])
    if currency != position.currency:
        raise ValueError(
            f"quote currency {currency!r} for {yahoo!r} disagrees with the position's "
            f"{position.currency!r} -- likely the wrong instrument; refusing to quote"
        )
    return Quote(
        symbol=yahoo,
        price=float(raw["price"]),
        currency=currency,
        as_of=datetime.fromtimestamp(int(raw["epoch"]), tz=timezone.utc),
        market_state=str(raw["market_state"]),
    )
