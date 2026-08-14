"""The reconciliation engine -- what the scanner says against what the camera sees.

The barcode says what the parcel should be. The camera says what is there. The
value is in the disagreement.

All four cells occur in the source clip; none are simulated:

    parcel   barcode        frames            meaning
    ------   ------------   ---------------   ------------------------------
    no       none           000-078, 287-391  idle station
    yes      TBA333...      150-240           expected parcel, readable label
    yes      wrong ('FSA')  410               unlabeled parcel -> DIVERT
    no       'FSA' present  420-460           detection miss, code still visible

Rows 1 and 3 are why this module exists. The barcode reports nothing in both --
one is a defect leaving the line, the other is a quiet Tuesday. Only the camera
separates them.

LIMITATIONS, both measured rather than assumed:

  - The verdict is per frame, not per parcel. A parcel visible across ~13
    sampled frames produces 13 independent verdicts. On the source clip that
    yields 6 DIVERTs of which 1 is a real defect: three while the label is still
    resolving, one single-frame decode dropout mid-dwell, one on departure. A
    per-parcel verdict is the fix.
  - APPROACHING cannot distinguish an arriving parcel from a departing one,
    because a stateless verdict has no memory of whether this parcel has already
    been inspected. It assumes an off-zone parcel is inbound, which holds on a
    continuously fed line.
"""

from __future__ import annotations

IDLE = "IDLE"
APPROACHING = "APPROACHING"
PASS = "PASS"
DIVERT = "DIVERT"
REVIEW = "REVIEW"


def decide(parcel_detected: bool, in_zone: bool, barcode: str | None) -> str:
    """Return a verdict for one frame.

    The order of the checks is the logic: each establishes a precondition for
    the next. A parcel exists, then it has arrived, then it has an identity.
    Checking the barcode before the zone would divert every parcel on its way in.
    """
    if not parcel_detected:
        return IDLE
    if not in_zone:
        return APPROACHING
    if barcode:
        return PASS
    return DIVERT
