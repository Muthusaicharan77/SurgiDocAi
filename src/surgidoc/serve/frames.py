"""Turn an uploaded video into the frame folder the pipeline reads.

The pipeline reads CholecSeg8k-style folders: frame_<n>_endo.png, where n is the
frame index in 25 fps footage (surgidoc.agent.perception). Uploaded videos can have
any frame rate, so frames are sampled at a fixed rate by timestamp and named by the
25 fps index of that timestamp. Times in the report then match the video's own
clock, and the run uses every extracted frame (every_n=1).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from surgidoc.agent.perception import FPS
from surgidoc.vision.dataset import IMAGE_SUFFIX

SAMPLE_FPS = 5  # the CLI default: every 5th frame of 25 fps footage
MAX_FRAMES = 3000  # 10 minutes at 5 fps; CPU perception takes ~1 s per frame


def extract_frames(video: Path, out_dir: Path, sample_fps: float = SAMPLE_FPS,
                   max_frames: int = MAX_FRAMES) -> int:
    """Decode video and write sampled frames to out_dir. Returns the number written."""
    import av

    out_dir.mkdir(parents=True, exist_ok=True)
    step = 1 / sample_fps
    next_t, written = 0.0, 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is None or frame.time + 1e-6 < next_t:
                continue
            if written >= max_frames:
                raise ValueError(f"video longer than {max_frames / sample_fps / 60:.0f} minutes")
            index = round(frame.time * FPS)
            image: Image.Image = frame.to_image()
            image.save(out_dir / f"frame_{index}{IMAGE_SUFFIX}")
            written += 1
            next_t += step
            while next_t <= frame.time:  # skip ahead after a gap in the stream
                next_t += step
    if written == 0:
        raise ValueError("no video frames could be decoded")
    return written
