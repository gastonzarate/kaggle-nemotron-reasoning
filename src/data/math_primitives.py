"""Grade-school arithmetic primitives that produce step-by-step explanations.

Each primitive returns a tuple (result, explanation_text). The explanation is a
human-readable walkthrough of the algorithm — column addition with carries,
column multiplication with partial products, long division with quotient digits,
etc. The text is designed so the model learns to FOLLOW the algorithm, not just
match the input/output pattern.

All arithmetic uses `Decimal` to avoid float surprises and to produce stable
2dp half-up rounding for the gravity/unit-conversion families.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Union

getcontext().prec = 50  # plenty of headroom for intermediate calculations

Number = Union[int, float, Decimal, str]


def D(n: Number) -> Decimal:
    """Coerce to Decimal cleanly. Float→str→Decimal avoids binary float drift."""
    if isinstance(n, Decimal):
        return n
    if isinstance(n, float):
        return Decimal(str(n))
    return Decimal(n)


def round_2dp(n: Number) -> Decimal:
    """Round half-up to 2 decimal places (matches the gold-answer convention)."""
    return D(n).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt(n: Decimal) -> str:
    """Format a Decimal cleanly: strip trailing zeros UNLESS that would drop the decimal point."""
    s = format(n, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return s


@dataclass
class ColumnAdd:
    """Adds two decimals column by column, showing carries.

    Example:
        ColumnAdd(D("12.5"), D("3.75")).result  → Decimal("16.25")
        ColumnAdd(D("12.5"), D("3.75")).explain →
            "    12.50
              +  3.75
              ──────
                16.25"
    """

    a: Decimal
    b: Decimal

    def __post_init__(self):
        self.result = self.a + self.b

    def explain(self) -> str:
        # Show two-line column addition with the result underlined.
        width = max(len(fmt(self.a)), len(fmt(self.b)), len(fmt(self.result))) + 2
        return (
            f"  {fmt(self.a).rjust(width)}\n"
            f"+ {fmt(self.b).rjust(width)}\n"
            f"  {'─' * width}\n"
            f"  {fmt(self.result).rjust(width)}"
        )


@dataclass
class ColumnSubtract:
    """Subtracts column by column, showing borrows in the result alignment."""

    a: Decimal
    b: Decimal

    def __post_init__(self):
        self.result = self.a - self.b

    def explain(self) -> str:
        width = max(len(fmt(self.a)), len(fmt(self.b)), len(fmt(self.result))) + 2
        return (
            f"  {fmt(self.a).rjust(width)}\n"
            f"- {fmt(self.b).rjust(width)}\n"
            f"  {'─' * width}\n"
            f"  {fmt(self.result).rjust(width)}"
        )


@dataclass
class ColumnMultiply:
    """Multiplies two decimals showing partial products digit-by-digit.

    Example:
        ColumnMultiply(D("4.62"), D("4.62")).explain →
            "    4.62
              x 4.62
              ──────
              0.0924    (4.62 * 0.02)
              2.7720    (4.62 * 0.60)
             18.4800    (4.62 * 4.00)
              ──────
             21.3444"
    """

    a: Decimal
    b: Decimal

    def __post_init__(self):
        self.result = self.a * self.b

    def explain(self) -> str:
        # Decompose b into its non-zero digit places, multiply each by a.
        # Example: b = 4.62 → 4.00, 0.60, 0.02
        partials = []
        b_str = fmt(self.b)
        if "." in b_str:
            int_part, dec_part = b_str.lstrip("-").split(".")
        else:
            int_part, dec_part = b_str.lstrip("-"), ""

        sign = -1 if self.b < 0 else 1

        # Decimal digits: index i in dec_part (left-to-right) → place 10^-(i+1)
        for i, d in enumerate(dec_part):
            if d != "0":
                place = D(d) * (Decimal(10) ** -(i + 1)) * sign
                partials.append((place, self.a * place))
        # Integer digits: index i in reversed int_part → place 10^i
        for i, d in enumerate(reversed(int_part)):
            if d != "0":
                place = D(d) * (Decimal(10) ** i) * sign
                partials.append((place, self.a * place))

        # Sort least-significant first for grade-school stair display
        partials.sort(key=lambda p: abs(p[0]))

        if not partials:
            return f"  {fmt(self.a)} × 0 = 0"

        width = max(len(fmt(p[1])) for p in partials)
        width = max(width, len(fmt(self.result))) + 2

        header = (
            f"  {fmt(self.a).rjust(width)}\n"
            f"× {fmt(self.b).rjust(width)}\n"
            f"  {'─' * width}"
        )
        lines = [header]
        for place, prod in partials:
            lines.append(f"  {fmt(prod).rjust(width)}    ({fmt(self.a)} × {fmt(place)})")
        lines.append(f"  {'─' * width}")
        lines.append(f"  {fmt(self.result).rjust(width)}")
        return "\n".join(lines)


@dataclass
class LongDivision:
    """Long division producing a quotient to N decimals.

    Walks digit-by-digit: for each position of the quotient, finds the digit d such
    that d * divisor ≤ current remainder, subtracts, and brings down the next digit.

    Example:
        LongDivision(D("177.99"), D("21.34"), decimals=2).result   → Decimal("8.34")
        LongDivision(D("177.99"), D("21.34"), decimals=2).explain →
            "Compute 177.99 ÷ 21.34:
               quotient digit 8 (units): 8 × 21.34 = 170.72; 177.99 - 170.72 = 7.27
               bring down → 72.70
               quotient digit 3 (tenths): 3 × 21.34 = 64.02; 72.70 - 64.02 = 8.68
               bring down → 86.80
               quotient digit 4 (hundredths): 4 × 21.34 = 85.36; 86.80 - 85.36 = 1.44
               Result: 8.34 (with remainder 0.0144)"
    """

    dividend: Decimal
    divisor: Decimal
    decimals: int = 2

    def __post_init__(self):
        if self.divisor == 0:
            raise ZeroDivisionError("divisor cannot be zero")
        self.result = (self.dividend / self.divisor).quantize(
            Decimal(10) ** -self.decimals, rounding=ROUND_HALF_UP
        )

    def explain(self) -> str:
        # Work with scaled integers to avoid decimal-point bookkeeping.
        # Scale both by 10^decimals so the divisor and dividend become integers.
        max_dec_dividend = -self.dividend.as_tuple().exponent if self.dividend.as_tuple().exponent < 0 else 0
        max_dec_divisor = -self.divisor.as_tuple().exponent if self.divisor.as_tuple().exponent < 0 else 0
        scale = max(max_dec_dividend, max_dec_divisor)
        N = int(self.dividend * (Decimal(10) ** scale))
        D_ = int(self.divisor * (Decimal(10) ** scale))

        lines = [f"Compute {fmt(self.dividend)} ÷ {fmt(self.divisor)}:"]

        # Integer part of the quotient
        int_quotient, remainder = divmod(N, D_)
        place_label = "units"
        lines.append(
            f"  integer part: {int_quotient} × {fmt(self.divisor)} = "
            f"{Decimal(int_quotient) * self.divisor}; remainder = "
            f"{Decimal(remainder) / (Decimal(10) ** scale)}"
        )

        # Decimal places
        for dec_i in range(self.decimals + 1):  # +1 for rounding
            remainder *= 10
            digit, remainder = divmod(remainder, D_)
            place_value = Decimal(digit) * (Decimal(10) ** -(dec_i + 1))
            place_label = ["tenths", "hundredths", "thousandths", "ten-thousandths"][min(dec_i, 3)]
            partial = D_ * digit
            lines.append(
                f"  next quotient digit: {digit} ({place_label}); "
                f"{digit} × {fmt(self.divisor)} = {Decimal(partial) / (Decimal(10) ** scale)}; "
                f"remainder = {Decimal(remainder) / (Decimal(10) ** scale)}"
            )

        lines.append(f"  → Result rounded to {self.decimals} decimals: {fmt(self.result)}")
        return "\n".join(lines)
