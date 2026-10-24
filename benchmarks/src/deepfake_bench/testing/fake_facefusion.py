"""Test double for `facefusion.py headless-run`, with the same CLI contract and no swapping.

It draws a filled ellipse over a fixed box ($FAKE_FACEFUSION_BOX = "x1,y1,x2,y2", default:
the centre of the frame) on every frame. The lab's coverage check therefore sees a change
in the face region, and no real face is ever altered.
$FAKE_FACEFUSION_MODE=noop copies the frames unchanged, to exercise coverage rejection.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import av
import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    args_in = sys.argv[1:] if argv is None else argv
    if args_in == ["--version"]:
        print("FakeFaceFusion 0.0 (test double)")
        return 0
    parser = argparse.ArgumentParser(prog="fake_facefusion")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("headless-run")
    run.add_argument("-s", "--source-paths", nargs="+", required=True)
    run.add_argument("-t", "--target-path", required=True)
    run.add_argument("-o", "--output-path", required=True)
    run.add_argument("--processors", nargs="+", required=True)
    run.add_argument("--face-swapper-model", required=True)
    args, _ = parser.parse_known_args(args_in)
    if not all(Path(p).is_file() for p in args.source_paths):
        print("source image missing", file=sys.stderr)
        return 2

    noop = os.environ.get("FAKE_FACEFUSION_MODE") == "noop"
    with av.open(args.target_path) as source, av.open(args.output_path, "w") as out:
        stream_in = source.streams.video[0]
        rate = stream_in.average_rate or 25
        stream = out.add_stream("libx264", rate=rate)
        stream.width = stream_in.codec_context.width
        stream.height = stream_in.codec_context.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        box = os.environ.get("FAKE_FACEFUSION_BOX")
        for index, frame in enumerate(source.decode(stream_in)):
            image = frame.to_ndarray(format="rgb24")
            if not noop:
                height, width = image.shape[:2]
                x1, y1, x2, y2 = (
                    [int(v) for v in box.split(",")]
                    if box
                    else [width // 4, height // 4, 3 * width // 4, 3 * height // 4]
                )
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
                shade = 40 + (index * 7) % 160
                cv2.ellipse(image, center, axes, 0, 0, 360, (shade, 90, 200 - shade // 2), -1)
            new = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            new.pts = index
            for packet in stream.encode(new):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
