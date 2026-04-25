import argparse
import math
import struct
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]


def matmul3(a: Mat3, b: Mat3) -> Mat3:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def matvec3(a: Mat3, v: Vec3) -> Vec3:
    return (
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    )


def transpose3(a: Mat3) -> Mat3:
    return (
        (a[0][0], a[1][0], a[2][0]),
        (a[0][1], a[1][1], a[2][1]),
        (a[0][2], a[1][2], a[2][2]),
    )


def det3(m: Mat3) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def normalize_quat(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    qw, qx, qy, qz = q
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0:
        return (1.0, 0.0, 0.0, 0.0)
    return (qw / n, qx / n, qy / n, qz / n)


def quat_to_rotmat(q: Tuple[float, float, float, float]) -> Mat3:
    qw, qx, qy, qz = normalize_quat(q)
    r00 = 1 - 2 * qy * qy - 2 * qz * qz
    r01 = 2 * qx * qy - 2 * qz * qw
    r02 = 2 * qx * qz + 2 * qy * qw

    r10 = 2 * qx * qy + 2 * qz * qw
    r11 = 1 - 2 * qx * qx - 2 * qz * qz
    r12 = 2 * qy * qz - 2 * qx * qw

    r20 = 2 * qx * qz - 2 * qy * qw
    r21 = 2 * qy * qz + 2 * qx * qw
    r22 = 1 - 2 * qx * qx - 2 * qy * qy
    return ((r00, r01, r02), (r10, r11, r12), (r20, r21, r22))


def rotmat_to_quat(r: Mat3) -> Tuple[float, float, float, float]:
    r00, r01, r02 = r[0]
    r10, r11, r12 = r[1]
    r20, r21, r22 = r[2]
    trace = r00 + r11 + r22

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif (r00 > r11) and (r00 > r22):
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s
    return normalize_quat((qw, qx, qy, qz))


def rot_x(deg: float) -> Mat3:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rot_y(deg: float) -> Mat3:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rot_z(deg: float) -> Mat3:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rot_zyx(yaw_z: float, pitch_y: float, roll_x: float) -> Mat3:
    return matmul3(matmul3(rot_z(yaw_z), rot_y(pitch_y)), rot_x(roll_x))


def parse_mat3(values: Sequence[float]) -> Mat3:
    if len(values) != 9:
        raise ValueError("Expected 9 numbers for a 3x3 matrix")
    v = list(map(float, values))
    return ((v[0], v[1], v[2]), (v[3], v[4], v[5]), (v[6], v[7], v[8]))


def preset_camera(name: str) -> Mat3:
    if name == "none":
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if name == "opengl_to_opencv":
        return ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    if name == "opencv_to_opengl":
        return ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    raise ValueError(f"Unknown camera preset: {name}")


def preset_world(name: str) -> Mat3:
    if name == "none":
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if name == "y_up_to_z_up":
        return ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    if name == "z_up_to_y_up":
        return ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    raise ValueError(f"Unknown world preset: {name}")


@dataclass
class ImageEntry:
    image_id: int
    q: Tuple[float, float, float, float]
    t: Vec3
    camera_id: int
    name: str
    points2d_line: str


def read_images_txt(path: Path) -> Tuple[List[str], List[ImageEntry]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    prefix: List[str] = []
    entries: List[ImageEntry] = []

    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i].strip() == ""):
        prefix.append(lines[i])
        i += 1

    while i < len(lines):
        if lines[i].startswith("#") or lines[i].strip() == "":
            prefix.append(lines[i])
            i += 1
            continue

        header = lines[i].strip()
        pts2d = lines[i + 1] if i + 1 < len(lines) else ""
        parts = header.split()
        if len(parts) < 10:
            raise ValueError(f"Bad images.txt header line: {header}")
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        entries.append(
            ImageEntry(
                image_id=image_id,
                q=(qw, qx, qy, qz),
                t=(tx, ty, tz),
                camera_id=camera_id,
                name=name,
                points2d_line=pts2d,
            )
        )
        i += 2

    return (prefix, entries)


def write_images_txt(path: Path, prefix: List[str], entries: List[ImageEntry]) -> None:
    out: List[str] = []
    out.extend(prefix)
    for e in entries:
        qw, qx, qy, qz = normalize_quat(e.q)
        tx, ty, tz = e.t
        out.append(
            f"{e.image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {e.camera_id} {e.name}"
        )
        out.append(e.points2d_line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def transform_images(entries: List[ImageEntry], camera_c: Mat3, world_g: Mat3) -> List[ImageEntry]:
    gt = transpose3(world_g)
    out: List[ImageEntry] = []
    for e in entries:
        r = quat_to_rotmat(e.q)
        r2 = matmul3(matmul3(camera_c, r), gt)
        t2 = matvec3(camera_c, e.t)
        out.append(
            ImageEntry(
                image_id=e.image_id,
                q=rotmat_to_quat(r2),
                t=t2,
                camera_id=e.camera_id,
                name=e.name,
                points2d_line=e.points2d_line,
            )
        )
    return out


def transform_points3d(in_path: Path, out_path: Path, world_g: Mat3) -> None:
    in_lines = in_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out_lines: List[str] = []
    for ln in in_lines:
        if ln.startswith("#") or ln.strip() == "":
            out_lines.append(ln)
            continue
        parts = ln.split()
        if len(parts) < 8:
            out_lines.append(ln)
            continue
        x, y, z = map(float, parts[1:4])
        x2, y2, z2 = matvec3(world_g, (x, y, z))
        parts[1] = f"{x2:.9f}"
        parts[2] = f"{y2:.9f}"
        parts[3] = f"{z2:.9f}"
        out_lines.append(" ".join(parts))
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def read_cameras_txt(path: Path) -> Dict[int, CameraModel]:
    cams: Dict[int, CameraModel] = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        cam_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = list(map(float, parts[4:]))
        if model == "PINHOLE":
            if len(params) < 4:
                raise ValueError(f"Bad PINHOLE params in cameras.txt for camera {cam_id}")
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model == "SIMPLE_PINHOLE":
            if len(params) < 3:
                raise ValueError(f"Bad SIMPLE_PINHOLE params in cameras.txt for camera {cam_id}")
            f, cx, cy = params[0], params[1], params[2]
            fx, fy = f, f
        else:
            raise ValueError(f"Unsupported camera model in cameras.txt: {model}")
        cams[cam_id] = CameraModel(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def write_cameras_txt(path: Path, cams: Dict[int, CameraModel]) -> None:
    out: List[str] = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cams)}",
    ]
    for cam_id in sorted(cams.keys()):
        cam = cams[cam_id]
        out.append(
            f"{cam_id} PINHOLE {cam.width} {cam.height} {cam.fx:.9f} {cam.fy:.9f} {cam.cx:.9f} {cam.cy:.9f}"
        )
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def rotate_image_intrinsics(cam: CameraModel, mode: str) -> CameraModel:
    w = cam.width
    h = cam.height
    if mode == "none":
        return cam
    if mode == "rot180":
        return CameraModel(
            width=w,
            height=h,
            fx=cam.fx,
            fy=cam.fy,
            cx=(w - 1.0) - cam.cx,
            cy=(h - 1.0) - cam.cy,
        )
    if mode == "cw90":
        return CameraModel(
            width=h,
            height=w,
            fx=cam.fy,
            fy=cam.fx,
            cx=(h - 1.0) - cam.cy,
            cy=cam.cx,
        )
    if mode == "ccw90":
        return CameraModel(
            width=h,
            height=w,
            fx=cam.fy,
            fy=cam.fx,
            cx=cam.cy,
            cy=(w - 1.0) - cam.cx,
        )
    raise ValueError(f"Unknown image rotation mode: {mode}")


def rotate_image_point(u: float, v: float, w: int, h: int, mode: str) -> Tuple[float, float]:
    if mode == "none":
        return (u, v)
    if mode == "rot180":
        return ((w - 1.0) - u, (h - 1.0) - v)
    if mode == "cw90":
        return ((h - 1.0) - v, u)
    if mode == "ccw90":
        return (v, (w - 1.0) - u)
    raise ValueError(f"Unknown image rotation mode: {mode}")


def rotate_points2d_line(line: str, cam: CameraModel, mode: str) -> str:
    if mode == "none" or not line.strip():
        return line
    parts = line.split()
    if len(parts) % 3 != 0:
        return line
    out: List[str] = []
    for i in range(0, len(parts), 3):
        u = float(parts[i])
        v = float(parts[i + 1])
        pid = parts[i + 2]
        u2, v2 = rotate_image_point(u, v, cam.width, cam.height, mode)
        out.append(f"{u2:.3f}")
        out.append(f"{v2:.3f}")
        out.append(pid)
    return " ".join(out)


def load_ply_xyz(path: Path) -> List[Vec3]:
    with path.open("rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header")
            header_bytes += line
            if line.strip() == b"end_header":
                break

        header = header_bytes.decode("ascii", errors="replace")
        if "format binary_little_endian" not in header:
            raise ValueError("Only binary_little_endian PLY supported")

        vertex_count: Optional[int] = None
        in_vertex = False
        props: List[Tuple[str, str]] = []
        for ln in header.splitlines():
            parts = ln.strip().split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append((parts[1], parts[2]))

        if vertex_count is None:
            raise ValueError("PLY header missing vertex count")
        if len(props) < 3 or props[0][1] != "x" or props[1][1] != "y" or props[2][1] != "z":
            raise ValueError(f"PLY vertex properties must start with x y z (got: {props[:6]})")

        ply_type_to_struct = {
            "char": "b",
            "int8": "b",
            "uchar": "B",
            "uint8": "B",
            "short": "h",
            "int16": "h",
            "ushort": "H",
            "uint16": "H",
            "int": "i",
            "int32": "i",
            "uint": "I",
            "uint32": "I",
            "float": "f",
            "float32": "f",
            "double": "d",
            "float64": "d",
        }

        fmt_parts: List[str] = []
        for t, _name in props:
            if t not in ply_type_to_struct:
                raise ValueError(f"Unsupported PLY property type: {t}")
            fmt_parts.append(ply_type_to_struct[t])

        fmt = "<" + "".join(fmt_parts)
        st = struct.Struct(fmt)
        stride = st.size
        data = f.read(vertex_count * stride)
        if len(data) != vertex_count * stride:
            raise ValueError("PLY vertex data truncated")

        pts: List[Vec3] = []
        mv = memoryview(data)
        for i in range(vertex_count):
            x, y, z = st.unpack_from(mv, i * stride)[:3]
            pts.append((float(x), float(y), float(z)))
        return pts


def jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        marker = data[i]
        i += 1
        if marker == 0xDA:
            break
        if marker in (0xD8, 0xD9):
            continue
        if i + 2 > len(data):
            break
        seglen = struct.unpack(">H", data[i : i + 2])[0]
        segstart = i + 2
        segend = segstart + seglen - 2
        if segend > len(data):
            break
        if marker in (0xC0, 0xC2):
            if segstart + 5 <= len(data):
                h = struct.unpack(">H", data[segstart + 1 : segstart + 3])[0]
                w = struct.unpack(">H", data[segstart + 3 : segstart + 5])[0]
                return (int(w), int(h))
        i = segend
    return None


def resolve_image_path(in_dir: Path, image_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    candidates.append(in_dir / image_name)
    if in_dir.parent:
        candidates.append(in_dir.parent / image_name)
    if in_dir.parent and in_dir.parent.parent:
        candidates.append(in_dir.parent.parent / "images" / image_name)
    if in_dir.parent and in_dir.parent.parent and in_dir.parent.parent.parent:
        root = in_dir.parent.parent.parent
        candidates.append(root / image_name)
        candidates.append(root / "images" / image_name)
        candidates.append(root / "COLMAP_Text_Model" / "images" / image_name)
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_image_rotate(mode: str, in_dir: Path, cams: Dict[int, CameraModel], images: List[ImageEntry]) -> str:
    if mode != "auto":
        return mode
    if not images:
        return "ccw90"
    e0 = images[0]
    cam = cams.get(e0.camera_id)
    if cam is None:
        return "ccw90"
    img_path = resolve_image_path(in_dir, e0.name)
    if img_path is None:
        return "ccw90"
    size = jpeg_size(img_path)
    if size is None:
        return "ccw90"
    iw, ih = size
    if cam.width == iw and cam.height == ih:
        return "none"
    return "ccw90"


def project(cam: CameraModel, r_w2c: Mat3, t_w2c: Vec3, xw: Vec3) -> Optional[Tuple[float, float, float]]:
    x, y, z = xw
    xc = r_w2c[0][0] * x + r_w2c[0][1] * y + r_w2c[0][2] * z + t_w2c[0]
    yc = r_w2c[1][0] * x + r_w2c[1][1] * y + r_w2c[1][2] * z + t_w2c[1]
    zc = r_w2c[2][0] * x + r_w2c[2][1] * y + r_w2c[2][2] * z + t_w2c[2]
    if zc <= 1e-6:
        return None
    u = cam.fx * (xc / zc) + cam.cx
    v = cam.fy * (yc / zc) + cam.cy
    if 0.0 <= u < float(cam.width) and 0.0 <= v < float(cam.height):
        return (u, v, zc)
    return None


def rebuild_from_ply(
    cameras: Dict[int, CameraModel],
    images: List[ImageEntry],
    ply_points_world: List[Vec3],
    max_points: int,
    max_views_per_point: int,
    min_views_per_point: int,
) -> Tuple[List[ImageEntry], List[str]]:
    img_points2d: Dict[int, List[Tuple[float, float, int]]] = {e.image_id: [] for e in images}
    points3d_lines: List[str] = []
    next_pid = 1

    for xw in ply_points_world:
        views: List[Tuple[float, int, float, float]] = []
        for e in images:
            cam = cameras.get(e.camera_id)
            if cam is None:
                continue
            r = quat_to_rotmat(e.q)
            p = project(cam, r, e.t, xw)
            if p is None:
                continue
            u, v, zc = p
            views.append((zc, e.image_id, u, v))

        if len(views) < min_views_per_point:
            continue

        views.sort(key=lambda t: t[0])
        views = views[:max_views_per_point]

        pid = next_pid
        next_pid += 1

        track_parts: List[str] = []
        for _zc, image_id, u, v in views:
            idx = len(img_points2d[image_id])
            img_points2d[image_id].append((u, v, pid))
            track_parts.append(str(image_id))
            track_parts.append(str(idx))

        x, y, z = xw
        points3d_lines.append(
            " ".join(
                [
                    str(pid),
                    f"{x:.9f}",
                    f"{y:.9f}",
                    f"{z:.9f}",
                    "255",
                    "255",
                    "255",
                    "1.000000",
                    *track_parts,
                ]
            )
        )

        if len(points3d_lines) >= max_points:
            break

    images_out: List[ImageEntry] = []
    for e in images:
        pts2d = img_points2d.get(e.image_id, [])
        if pts2d:
            parts: List[str] = []
            for u, v, pid in pts2d:
                parts.append(f"{u:.3f}")
                parts.append(f"{v:.3f}")
                parts.append(str(pid))
            pts2d_line = " ".join(parts)
        else:
            pts2d_line = ""
        images_out.append(
            ImageEntry(
                image_id=e.image_id,
                q=e.q,
                t=e.t,
                camera_id=e.camera_id,
                name=e.name,
                points2d_line=pts2d_line,
            )
        )

    return (images_out, points3d_lines)


def read_points_xyz(path: Path, max_n: int = 20000) -> List[Vec3]:
    pts: List[Vec3] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            pts.append((x, y, z))
            if len(pts) >= max_n:
                break
    return pts


def mean_point(pts: Sequence[Vec3]) -> Vec3:
    sx = sy = sz = 0.0
    for x, y, z in pts:
        sx += x
        sy += y
        sz += z
    n = len(pts)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (sx / n, sy / n, sz / n)


def norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize(v: Vec3) -> Vec3:
    n = norm(v)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def camera_center_world(r_w2c: Mat3, t_w2c: Vec3) -> Vec3:
    rt = transpose3(r_w2c)
    return matvec3(rt, (-t_w2c[0], -t_w2c[1], -t_w2c[2]))


def camera_forward_world(r_w2c: Mat3) -> Vec3:
    rt = transpose3(r_w2c)
    return (rt[0][2], rt[1][2], rt[2][2])


def check_forward(model_dir: Path) -> None:
    images_path = model_dir / "images.txt"
    points_path = model_dir / "points3D.txt"
    _, entries = read_images_txt(images_path)
    pts = read_points_xyz(points_path)
    cent = mean_point(pts)
    vals: List[float] = []
    for e in entries:
        r = quat_to_rotmat(e.q)
        c = camera_center_world(r, e.t)
        fwd = camera_forward_world(r)
        to_cent = normalize((cent[0] - c[0], cent[1] - c[1], cent[2] - c[2]))
        vals.append(dot(fwd, to_cent))
    vals.sort()
    if vals:
        print("centroid:", f"{cent[0]:.6f}", f"{cent[1]:.6f}", f"{cent[2]:.6f}")
        print("dot(forward, to_centroid):", "min", f"{vals[0]:.6f}", "median", f"{vals[len(vals)//2]:.6f}", "max", f"{vals[-1]:.6f}")
    else:
        print("No points found for forward-check.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="in_dir",
        required=True,
        help="Path to COLMAP text model directory (contains cameras.txt/images.txt/points3D.txt)",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        default=None,
        help="Output directory for converted COLMAP text model (default: overwrite input directory)",
    )
    p.add_argument("--overwrite", action="store_true", default=True)

    p.add_argument(
        "--camera-preset",
        default="none",
        choices=["none", "opengl_to_opencv", "opencv_to_opengl"],
        help="Camera-axis convention transform applied to COLMAP extrinsics (default: none). Use OpenGL<->OpenCV only if your source/target camera axes differ.",
    )
    p.add_argument("--world-preset", default="y_up_to_z_up", choices=["none", "y_up_to_z_up", "z_up_to_y_up"])

    p.add_argument("--camera-euler-zyx-deg", nargs=3, type=float, metavar=("YAW_Z", "PITCH_Y", "ROLL_X"))
    p.add_argument("--world-euler-zyx-deg", nargs=3, type=float, metavar=("YAW_Z", "PITCH_Y", "ROLL_X"))

    p.add_argument("--camera-mat3", nargs=9, type=float)
    p.add_argument("--world-mat3", nargs=9, type=float)

    p.add_argument("--image-rotate", default="auto", choices=["auto", "none", "cw90", "ccw90", "rot180"])

    p.add_argument("--ply", help="Optional LiDAR point cloud PLY to (re)generate points3D.txt + POINTS2D tracks")
    p.add_argument("--max-points", type=int, default=20000)
    p.add_argument("--max-views-per-point", type=int, default=3)
    p.add_argument("--min-views-per-point", type=int, default=2)

    p.add_argument("--check-forward", action="store_true")
    args = p.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    in_cameras = in_dir / "cameras.txt"
    in_images = in_dir / "images.txt"
    in_points = in_dir / "points3D.txt"
    for fp in (in_cameras, in_images, in_points):
        if not fp.exists():
            raise SystemExit(f"Missing required file: {fp}")

    if out_dir.exists() and (out_dir.resolve() != in_dir.resolve()):
        if not args.overwrite:
            raise SystemExit(f"Output directory exists. Use --overwrite to replace: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_c = preset_camera(args.camera_preset)
    world_g = preset_world(args.world_preset)

    if args.camera_euler_zyx_deg is not None:
        y, pch, r = args.camera_euler_zyx_deg
        camera_c = matmul3(rot_zyx(y, pch, r), camera_c)
    if args.world_euler_zyx_deg is not None:
        y, pch, r = args.world_euler_zyx_deg
        world_g = matmul3(rot_zyx(y, pch, r), world_g)

    if args.camera_mat3 is not None:
        camera_c = matmul3(parse_mat3(args.camera_mat3), camera_c)
    if args.world_mat3 is not None:
        world_g = matmul3(parse_mat3(args.world_mat3), world_g)

    if abs(det3(camera_c) - 1.0) > 1e-4:
        raise SystemExit(f"camera transform det not ~1 (got {det3(camera_c):.6f}). Need a proper rotation.")
    if abs(det3(world_g) - 1.0) > 1e-4:
        raise SystemExit(f"world transform det not ~1 (got {det3(world_g):.6f}). Need a proper rotation.")

    cams_in = read_cameras_txt(in_cameras)
    prefix, images = read_images_txt(in_images)
    effective_image_rotate = resolve_image_rotate(args.image_rotate, in_dir, cams_in, images)

    if effective_image_rotate == "cw90":
        camera_c = matmul3(rot_z(-90.0), camera_c)
    elif effective_image_rotate == "ccw90":
        camera_c = matmul3(rot_z(90.0), camera_c)
    elif effective_image_rotate == "rot180":
        camera_c = matmul3(rot_z(180.0), camera_c)

    cams_out: Dict[int, CameraModel] = {cid: rotate_image_intrinsics(cam, effective_image_rotate) for cid, cam in cams_in.items()}
    write_cameras_txt(out_dir / "cameras.txt", cams_out)

    images2 = transform_images(images, camera_c=camera_c, world_g=world_g)

    images2 = [
        ImageEntry(
            image_id=e.image_id,
            q=e.q,
            t=e.t,
            camera_id=e.camera_id,
            name=e.name,
            points2d_line=rotate_points2d_line(
                e.points2d_line,
                cams_in.get(e.camera_id, CameraModel(0, 0, 0, 0, 0, 0)),
                effective_image_rotate,
            ),
        )
        for e in images2
    ]

    if args.ply:
        ply_path = Path(args.ply)
        if not ply_path.exists():
            raise SystemExit(f"PLY not found: {ply_path}")
        ply_points = [matvec3(world_g, p3) for p3 in load_ply_xyz(ply_path)]
        images3, points3d_lines = rebuild_from_ply(
            cameras=cams_out,
            images=images2,
            ply_points_world=ply_points,
            max_points=max(1, int(args.max_points)),
            max_views_per_point=max(1, int(args.max_views_per_point)),
            min_views_per_point=max(1, int(args.min_views_per_point)),
        )

        write_images_txt(out_dir / "images.txt", prefix=prefix, entries=images3)

        mean_track = 0.0
        if points3d_lines:
            mean_track = sum((len(ln.split()) - 8) / 2.0 for ln in points3d_lines) / float(len(points3d_lines))

        pts_out = [
            "# 3D point list with one line of data per point:",
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
            f"# Number of points: {len(points3d_lines)}, mean track length: {mean_track:.3f}",
            *points3d_lines,
        ]
        (out_dir / "points3D.txt").write_text("\n".join(pts_out) + "\n", encoding="utf-8")
    else:
        write_images_txt(out_dir / "images.txt", prefix=prefix, entries=images2)
        transform_points3d(in_points, out_dir / "points3D.txt", world_g=world_g)

    print("Wrote converted COLMAP model to:", str(out_dir))
    if args.check_forward:
        check_forward(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

