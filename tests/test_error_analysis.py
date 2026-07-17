from visionguard.error_analysis import Box, box_iou


def test_box_iou_handles_overlap_and_disjoint_boxes() -> None:
    first = Box(0, (0.0, 0.0, 10.0, 10.0))
    overlap = Box(0, (5.0, 5.0, 15.0, 15.0))
    disjoint = Box(0, (20.0, 20.0, 30.0, 30.0))

    assert box_iou(first, overlap) == 25 / 175
    assert box_iou(first, disjoint) == 0.0
