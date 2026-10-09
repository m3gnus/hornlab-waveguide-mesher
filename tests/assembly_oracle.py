"""Independent finite-surface oracle for shared assembly test meshes."""

import math

import numpy as np


def line(a, b):
    ar, az = a
    br, bz = b
    dr, dz = br - ar, bz - az

    def distance(r, z):
        t = np.clip(((r - ar) * dr + (z - az) * dz) / (dr * dr + dz * dz), 0, 1)
        return np.hypot(r - ar - t * dr, z - az - t * dz)

    return distance


def segment(model, i):
    a, b = model.points[i : i + 2]
    s = model.segments[i]
    if s.kind == "line":
        return line((a.r_mm, a.z_mm), (b.r_mm, b.z_mm))
    cr, cz = s.center_mm
    radius = math.hypot(a.r_mm - cr, a.z_mm - cz)
    start = math.atan2(a.z_mm - cz, a.r_mm - cr)
    end = math.atan2(b.z_mm - cz, b.r_mm - cr)
    angle = (end - start) % (2 * math.pi)
    sweep = angle if s.direction == "ccw" else angle - 2 * math.pi

    def distance(r, z):
        theta = np.arctan2(z - cz, r - cr)
        progress = np.mod((theta - start) * math.copysign(1, sweep), 2 * math.pi)
        progress = np.where(abs(progress - 2 * math.pi) < 1e-10, 0, progress)
        ends = np.minimum(
            np.hypot(r - a.r_mm, z - a.z_mm), np.hypot(r - b.r_mm, z - b.z_mm)
        )
        return np.where(
            progress <= abs(sweep) + 1e-10, abs(np.hypot(r - cr, z - cz) - radius), ends
        )

    return distance


def certificate_xyz(facets, distance):
    worst = 0.0
    for n in (32, 128, 256):
        uv = np.array([(a / n, b / n) for a in range(n + 1) for b in range(n + 1 - a)])
        pending = []
        for batch in np.array_split(facets, max(1, len(facets) // 24 + 1)):
            samples = (
                batch[:, 0, None, :]
                + uv[None, :, 0, None] * (batch[:, 1, None, :] - batch[:, 0, None, :])
                + uv[None, :, 1, None] * (batch[:, 2, None, :] - batch[:, 0, None, :])
            )
            sampled = distance(samples).max(axis=1)
            assert sampled.max() <= 0.15, sampled.max()
            bound = (
                sampled
                + np.linalg.norm(batch - np.roll(batch, 1, axis=1), axis=2).max(axis=1)
                / n
            )
            good = bound <= 0.15
            if good.any():
                worst = max(worst, float(bound[good].max()))
            pending.extend(batch[~good])
        if not pending:
            return worst
        facets = np.asarray(pending)
    raise AssertionError("whole-facet bound could not be certified")


def independent_surfaces(model):
    def radial(origin, fn):
        def distance(p):
            q = p - np.asarray(origin)
            return fn(np.hypot(q[..., 0], q[..., 1]), q[..., 2])

        return distance

    moving, rigid = {}, []
    for key, c, i, origin, _ in model.patches:
        distance = radial(origin, segment(c, i))
        if c.segments[i].role == "moving":
            moving[key] = distance
        else:
            rigid.append(distance)
    z, w, h, d = (
        model.front_z_mm,
        model.width_mm / 2,
        model.height_mm / 2,
        model.depth_mm,
    )

    def rectangle(p, axis, plane, a, b):
        others = [i for i in range(3) if i != axis]
        return np.sqrt(
            (p[..., axis] - plane) ** 2
            + sum(
                np.maximum(np.maximum(low - p[..., k], p[..., k] - high), 0) ** 2
                for k, (low, high) in zip(others, [a, b])
            )
        )

    def front(p):
        result = rectangle(p, 2, z, (-w, w), (-h, h))
        for center, radius in (
            (model.horn_xy_mm, model.mouth_radius_mm),
            (model.woofer_xy_mm, model.aperture_radius_mm),
        ):
            r = np.hypot(p[..., 0] - center[0], p[..., 1] - center[1])
            result = np.where(r < radius, np.hypot(p[..., 2] - z, radius - r), result)
        return result

    rigid += [
        front,
        lambda p: rectangle(p, 2, z - d, (-w, w), (-h, h)),
        lambda p: rectangle(p, 0, -w, (-h, h), (z - d, z)),
        lambda p: rectangle(p, 0, w, (-h, h), (z - d, z)),
        lambda p: rectangle(p, 1, -h, (-w, w), (z - d, z)),
        lambda p: rectangle(p, 1, h, (-w, w), (z - d, z)),
        radial(
            model.parts[0][1],
            line(
                (model.horn.points[-1].r_mm, 0),
                (model.mouth_radius_mm, model.horn_length_mm),
            ),
        ),
    ]
    if model.aperture_radius_mm > model.woofer.points[-1].r_mm:
        rigid.append(
            radial(
                model.parts[1][1],
                line((model.woofer.points[-1].r_mm, 0), (model.aperture_radius_mm, 0)),
            )
        )
    return moving, lambda p: np.minimum.reduce([f(p) for f in rigid])
