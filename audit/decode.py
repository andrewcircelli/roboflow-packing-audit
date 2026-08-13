"""Barcode decode -- the label side of the correlation.

Measured constraints this module works within:

  - The shipping label decodes as CODE128 'TBA333622790661', rect 407x121.
  - The mailer carries its OWN CODE128, 'FSA', rect ~311x627, visible when the
    parcel is flipped -- that is, in exactly the frames that must fail. Both are
    CODE128, so symbology cannot separate them. Both are physically on the
    parcel, so location cannot either.
  - Decode needs ~440-500 px along the barcode's long axis. It works at 1920px
    frame width and fails at 1280px.
  - Motion blur defeats decode outright while detection tolerates it: the same
    2160x3840 source gave 0.446 detection with zero decodes while the parcel was
    moving, and 0.970 with clean decodes at rest.

The whole frame is scanned rather than a crop of the detection. Cropping would
be faster and would ignore barcodes elsewhere in the scene, but it would make
decode depend on detection -- and the architecture rests on the two signals
failing independently. Survey frames 00443 and 00495 had a readable barcode and
no detection at all; a crop-to-detection decode would be blind to exactly those.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from pyzbar.pyzbar import ZBarSymbol
from pyzbar.pyzbar import decode as zbar_decode

# Carrier prefixes that identify a parcel's tracking barcode.
#
# LIMITATION: 'TBA' is Amazon's prefix. A customer shipping UPS or FedEx needs
# this list extended -- this is per-deployment configuration, and the first
# thing to check when a new carrier appears on the line.
VALID_PREFIXES = ("TBA",)


def read_barcode(
    frame: np.ndarray, detections: sv.Detections, polygon: np.ndarray
) -> str | None:
    """Return the parcel's tracking identifier, or None if there isn't one.

    Every CODE128 in the frame is checked rather than only the first zbar
    returns. When both the shipping label and the mailer's own 'FSA' marking are
    visible, taking the first result is a coin flip.

    Returning None for a barcode that decoded but failed validation is
    deliberate: an unrecognised code means the parcel has no identity this
    station can act on, which is operationally the same as no code at all.
    """
    grayscale = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for barcode in zbar_decode(grayscale, symbols=[ZBarSymbol.CODE128]):
        text = barcode.data.decode("utf-8", errors="replace")
        if text.startswith(VALID_PREFIXES):
            return text
    return None
