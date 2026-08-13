"""The checkpoint region -- the stationary scanner's read window.

The zone is calibrated from a survey of the source clip (20 frames, 2160x3840,
frame centre x=1080). cx is the centre of the detected parcel:

    frame_00104   cx= 734   54% of frame   entering, off-centre
    frame_00130   cx=1073   83%            arriving
    frame_00156   cx=1080   86%            settled   <- barcode decodes
    frame_00182   cx=1080   87%            settled   <- barcode decodes
    frame_00208   cx=1080   91%            settled   <- barcode decodes
    frame_00234   cx=1080   93%            settled   <- barcode decodes
    frame_00261   cx=1343   71%            leaving, off-centre

The barcode decoded only inside the settled window. The zone therefore predicts
whether the label is readable -- it is the scanner's read window, not a
state-machine flourish.

cx separates the states cleanly; cy does not (settled 1992-2172, in transit
2035-2307, heavily overlapped). The zone constrains x only. Constraining a
dimension that does not discriminate adds a way to fail without adding a way to
be right.

LIMITATION: the polygon is in pixel coordinates, so it is bound to this exact
camera position. Moving or bumping the camera requires re-deriving cx from a new
survey. See README, "Re-calibration".
"""

from __future__ import annotations

import numpy as np
import supervision as sv

# Half-width of the read window either side of frame centre, in pixels.
# 150 sits between the settled cluster (cx=1080) and the nearest reject
# (cx=1343, 263px away), with margin on both sides.
ZONE_HALF_WIDTH_PX = 150


def build_zone(frame_width: int, frame_height: int) -> np.ndarray:
    """Return the checkpoint polygon as an (N, 2) array of pixel coordinates.

    A vertical band spanning the full frame height. Corners are listed in order
    around the rectangle; listing them diagonally would produce a bowtie and
    break the point-in-polygon test.
    """
    centre_x = frame_width // 2
    left = centre_x - ZONE_HALF_WIDTH_PX
    right = centre_x + ZONE_HALF_WIDTH_PX
    return np.array(
        [[left, 0], [right, 0], [right, frame_height], [left, frame_height]]
    )


def is_in_zone(
    detections: sv.Detections, polygon: np.ndarray, frame_size: tuple[int, int]
) -> bool:
    """True when the parcel is at the read position.

    The anchor is CENTER rather than supervision's BOTTOM_CENTER default,
    because the zone was calibrated against the parcel's centre point. A
    different anchor would test a different point and the 150px half-width
    would not mean what it was derived to mean.

    The parcel covers 86-93% of the frame when settled, so it never fits inside
    the band. The band is a target for its centre, not a container for it.
    """
    zone = sv.PolygonZone(polygon=polygon, triggering_anchors=(sv.Position.CENTER,))
    return bool(zone.trigger(detections).any())
