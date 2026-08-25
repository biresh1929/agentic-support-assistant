"""The only monetary figures the assistant may state, and when.

Deliberately a hard-coded allowlist rather than a classifier or a fuzzy match
against retrieved text. Four figures exist in the policy; each is tied to the
single condition that licenses it. Anything else -- a percentage off, a
goodwill credit, a refund total the model computed itself -- is blocked,
whatever justification surrounds it. This is meant to be walked through line
by line and defended.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AllowedAmount:
    rupees: int
    citation: str
    condition: str

    def describe(self) -> str:
        return f"₹{self.rupees} ({self.citation}) — only when {self.condition}"


# policy 1.5, 5.2, 2.5, 3.2 respectively.
DELAY_CREDIT = AllowedAmount(
    rupees=250,
    citation="Shipping -> 1.5 Delayed orders",
    condition="an order is more than 3 business days past expected delivery",
)
COURIER_REIMBURSEMENT = AllowedAmount(
    rupees=150,
    citation="Return pickup -> 5.2 Non-serviceable pincodes",
    condition="the customer self-ships from a non-serviceable pincode, against a receipt",
)
FOOTWEAR_BOX_DEDUCTION = AllowedAmount(
    rupees=300,
    citation="Returns -> 2.5 Footwear",
    condition="footwear is returned without its original shoe box",
)
SHIPPING_FEE_REFUND = AllowedAmount(
    rupees=99,
    citation="Refunds -> 3.2 Shipping fees",
    condition="the return is due to a Trendly error (wrong, damaged or defective item)",
)

ALLOWED_AMOUNTS: tuple[AllowedAmount, ...] = (
    DELAY_CREDIT,
    COURIER_REIMBURSEMENT,
    FOOTWEAR_BOX_DEDUCTION,
    SHIPPING_FEE_REFUND,
)

ALLOWED_RUPEE_VALUES: frozenset[int] = frozenset(a.rupees for a in ALLOWED_AMOUNTS)

# Figures that are policy facts but not payable amounts, so a bare number in an
# answer is not automatically a violation. Kept separate from ALLOWED_AMOUNTS
# because these must never be described as money owed to a customer.
POLICY_THRESHOLDS: frozenset[int] = frozenset(
    {
        1499,  # 1.3 free-shipping threshold
        199,   # 1.3 express shipping charge
        30,    # 2.1 return window, days
        48,    # 6.1 damage reporting window, hours
        10,    # 1.2 remote pincode days / 1.6 no-movement days
    }
)
