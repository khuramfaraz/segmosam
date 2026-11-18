import numpy as np
import segmosam


def main():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    box = (10, 12, 16, 14)

    tracker = segmosam.TrackerEngine("", 10)
    tracker.init_target(frame, box)
    mask, latency_ms = tracker.process_frame(frame)

    expected_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    expected_mask[box[1] : box[1] + box[3], box[0] : box[0] + box[2]] = 255

    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
    assert latency_ms >= 0.0
    np.testing.assert_array_equal(mask, expected_mask)
    print("segmosam smoke test passed")


if __name__ == "__main__":
    main()