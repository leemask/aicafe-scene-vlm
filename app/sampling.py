"""Label-blind, bounded frame selection. Motion scores are not person detections."""
import math
from PIL import ImageChops, ImageFilter, ImageStat

SAMPLER_VERSION = "motion-v1"


def uniform_times(start, end, count):
    if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end <= start:
        raise ValueError("Invalid interval")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 16:
        raise ValueError("Frame budget must be between 1 and 16")
    return [start + (end - start) * (i + .5) / count for i in range(count)]


def decode_times(path, times, end):
    """Return first decoded frame at/after each target, with actual unique PTS."""
    import av
    result = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for target in sorted(set(times)):
            container.seek(int(target / float(stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                actual = float(frame.pts * stream.time_base)
                if actual + 1e-6 < target:
                    continue
                if actual >= end:
                    break
                if not result or actual > result[-1][0] + 1e-6:
                    image = frame.to_image()
                    image.thumbnail((960, 540))
                    result.append((actual, image))
                break
    if not result:
        raise ValueError("No decoded frames in interval")
    return result


def motion_probes(path, start, end, probe_hz=2):
    """Streaming decode; retain one small previous image and scalar scores."""
    import av
    uniform_times(start, end, 1)
    if not math.isfinite(probe_hz) or not .5 <= probe_hz <= 8:
        raise ValueError("Probe rate must be between .5 and 8 Hz")
    if (end - start) * probe_hz > 4800:
        raise ValueError("Probe budget exceeded")
    previous = None
    probes = []
    target = start
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / float(stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * stream.time_base)
            if actual >= end:
                break
            if actual + 1e-6 < target:
                continue
            current = frame.to_image().convert("L").resize((160, 96)).filter(ImageFilter.BoxBlur(1))
            score = 0.
            if previous is not None:
                difference = ImageChops.difference(previous, current).point(lambda x: x if x >= 12 else 0)
                # Score local motion using the maximum tile mean.
                score = max(
                    ImageStat.Stat(difference.crop((x, y, x+40, y+24))).mean[0]
                    for x in range(0, 160, 40) for y in range(0, 96, 24)
                ) / 255.
            probes.append({"time": actual, "score": score})
            previous = current
            target = start + (math.floor((actual-start)*probe_hz + 1e-6)+1)/probe_hz
    return probes


def hybrid_times(start, end, count, probes):
    """Keep uniform anchors; add separated motion peaks, then gap fillers."""
    uniform_times(start, end, count)
    selected = uniform_times(start, end, (count + 1)//2)
    spacing = (end-start)/count*.35
    candidates = [
        p for p in probes
        if math.isfinite(p["time"]) and math.isfinite(p["score"])
        and start <= p["time"] < end
    ]
    for p in sorted(candidates, key=lambda p: (-p["score"], p["time"])):
        if len(selected) >= count:
            break
        if p["score"] >= .005 and min(abs(p["time"]-t) for t in selected) >= spacing:
            selected.append(p["time"])
    # Quiet scenes still receive temporal coverage; low motion never means no person.
    remaining = sorted(set([p["time"] for p in candidates] + uniform_times(start, end, count)))
    while len(selected) < count:
        choices = [t for t in remaining if all(abs(t-s)>1e-6 for s in selected)]
        if not choices:
            break
        best = max(choices, key=lambda t: (min(abs(t-s) for s in selected), -t))
        selected.append(best)
    return sorted(selected)


def sample_frames(path, start, end, count=6, strategy="uniform", probes=None):
    targets = uniform_times(start, end, count)
    if strategy == "hybrid":
        if probes is None:
            probes = motion_probes(path, start, end)
        targets = hybrid_times(start, end, count, probes)
    elif strategy != "uniform":
        raise ValueError("Unknown sampling strategy")
    return decode_times(path, targets, end)
