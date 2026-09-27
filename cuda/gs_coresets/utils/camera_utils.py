# camera_utils.py (PyTorch device-aware, elliptical splats, correct background blending)
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import torch
import numpy as np
from typing import Any
from gs_coresets.utils.types_devices_utils import as_4x4_np, torch_ctx


def normalize_vector(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize vectors along the last dimension with numerical guarding."""
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps))


def look_at_c2w(eye: torch.Tensor,
                target: torch.Tensor,
                up_hint: torch.Tensor) -> torch.Tensor:
    """Create a right-handed camera-to-world matrix using a look-at convention."""
    f = normalize_vector(target - eye)                         # forward (+Z)
    r = torch.linalg.cross(f, up_hint)                         # provisional right
    if torch.linalg.norm(r) < 1e-6:
        alt_up = torch.tensor([1.0, 0.0, 0.0], device=eye.device, dtype=eye.dtype)
        r = torch.linalg.cross(f, alt_up)
    r = normalize_vector(r)                                    # final right (+X)
    u = torch.linalg.cross(r, f)                               # final up (+Y)
    c2w = torch.eye(4, device=eye.device, dtype=eye.dtype)
    c2w[:3, 0] = r
    c2w[:3, 1] = u
    c2w[:3, 2] = f
    c2w[:3, 3] = eye
    return c2w


@dataclass
class Camera:
    width: int
    height: int
    world_view_transform: torch.Tensor  # 4x4 (world->camera)
    projection_matrix: Optional[torch.Tensor] = None
    fx: float = 0.0
    fy: float = 0.0
    FoVx: float = 0.0
    FoVy: float = 0.0
    camera_center: Optional[torch.Tensor] = None  # (3,)

    @staticmethod
    def from_view_fov(viewmatrix: torch.Tensor,
                      image_width: int,
                      image_height: int,
                      fov_x: float,
                      fov_y: float,
                      near: float = 0.1,
                      far: float = 1000.0) -> "Camera":
        fx = (image_width / 2.0) / math.tan(fov_x / 2.0)
        fy = (image_height / 2.0) / math.tan(fov_y / 2.0)
        return Camera(
            width=image_width,
            height=image_height,
            world_view_transform=viewmatrix.to(torch_ctx.dtype),
            projection_matrix=None,
            fx=float(fx),
            fy=float(fy),
            FoVx=float(fov_x),
            FoVy=float(fov_y),
            camera_center=None,
        )



def _project_points(means3d_cam: torch.Tensor, fx: float, fy: float, W: int, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X, Y, Z = means3d_cam.unbind(-1)
    invZ = 1.0 / (Z + 1e-8)
    u = fx * (X * invZ) + (W * 0.5)
    v = fy * (Y * invZ) + (H * 0.5)
    return u, v

def _jacobian_pinhole(Xc: torch.Tensor, fx: float, fy: float,
                      fovx: float | None = None, fovy: float | None = None,
                      clamp_factor: float = 1.3) -> torch.Tensor:
    """
    (N,3)->(N,2,3) Jacobian of (u,v) wrt (X,Y,Z), OpenCV/COLMAP (v-down).
    If FoVs are given, clamp x/z and y/z to +-clamp_factor * tan(FoV/2)
    before building the d/dZ terms (matches GraphDECO CUDA stabilizer).
    """
    X, Y, Z = Xc.unbind(-1)
    invZ  = 1.0 / (Z + 1e-8)
    invZ2 = invZ * invZ

    if (fovx is not None) and (fovy is not None):
        # normalized slopes
        x_over_z = X * invZ
        y_over_z = Y * invZ
        # clamp ranges
        tx = math.tan(fovx * 0.5) * clamp_factor
        ty = math.tan(fovy * 0.5) * clamp_factor
        x_over_z = torch.clamp(x_over_z, -tx, tx)
        y_over_z = torch.clamp(y_over_z, -ty, ty)
        # Jacobian
        J = torch.zeros(Xc.shape[0], 2, 3, device=Xc.device, dtype=Xc.dtype)
        J[:, 0, 0] = fx * invZ
        J[:, 0, 2] = -fx * x_over_z * invZ         # NOTE: clamped slope used here
        J[:, 1, 1] = fy * invZ
        J[:, 1, 2] = -fy * y_over_z * invZ         # NOTE: clamped slope used here
        return J

    # Fallback (no clamp)
    J = torch.zeros(Xc.shape[0], 2, 3, device=Xc.device, dtype=Xc.dtype)
    J[:, 0, 0] = fx * invZ
    J[:, 0, 2] = -fx * X * invZ2
    J[:, 1, 1] = fy * invZ
    J[:, 1, 2] = -fy * Y * invZ2
    return J


def _invert_2x2(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = S[:, 0, 0]; b = S[:, 0, 1]; c = S[:, 1, 1]
    det = a * c - b * b
    det = torch.clamp(det, min=eps)
    inv = torch.zeros_like(S)
    inv[:, 0, 0] = c / det
    inv[:, 1, 1] = a / det
    minus_b_over_det = -b / det
    inv[:, 0, 1] = minus_b_over_det
    inv[:, 1, 0] = minus_b_over_det
    return inv


def projected_covariance_2d_batched(
    *,
    means3d: torch.Tensor,          # (N,3) in WORLD
    scales_log: torch.Tensor,       # (N,3) GraphDECO log-scales
    rotations_wxyz: torch.Tensor,   # (N,4) WXYZ
    cameras: List["Camera"],        # B cameras (must share W,H)
    clamp_factor: float = 1.3,
    return_sigmas: bool = False,
    min_sigma_px: float = 0.5,
    max_sigma_px: Optional[float] = None,
    is_render = True,
) -> Dict[str, torch.Tensor]:
    """
    GraphDECO-correct Σ2D for B cameras in one vectorized pass:
      Σ_w = R_g diag(exp(log_s)^2) R_g^T
      Σ_c = R_c Σ_w R_c^T
      J uses clamped slopes: 1.3 * tan(FoV/2) (numerical stabilizer)
      Σ_2d = J Σ_c J^T, and we return Σ_2d^{-1} (Sinv) + centers (u,v) + depth Z.

    Returns:
      {"Sinv": (B,N,2,2), "uc": (B,N), "vc": (B,N), "Z": (B,N)}  [+ "sigmas": (B,N,2) if requested]
    """
    dev, dtype = means3d.device, means3d.dtype
    B = len(cameras)
    N = means3d.shape[0]
    if B == 0:
        raise ValueError("cameras must be a non-empty list")

    # ----- world Gaussian covariance: Σ_w (N,3,3)
    s = torch.exp(scales_log).to(dtype=dtype, device=dev)          # (N,3)
    s2 = s * s
    Rg = quat_wxyz_to_rotmat(rotations_wxyz.to(dev, dtype), is_render)       # (N,3,3)
    Sw = Rg @ torch.diag_embed(s2) @ Rg.transpose(1, 2)            # (N,3,3)

    # ----- stack camera parameters
    W = cameras[0].width; H = cameras[0].height
    if not all(c.width == W and c.height == H for c in cameras):
        raise AssertionError("All cameras must share W,H for the batched path.")
    fx = torch.tensor([c.fx   for c in cameras], device=dev, dtype=dtype)   # (B,)
    fy = torch.tensor([c.fy   for c in cameras], device=dev, dtype=dtype)
    fovx = torch.tensor([c.FoVx for c in cameras], device=dev, dtype=dtype)
    fovy = torch.tensor([c.FoVy for c in cameras], device=dev, dtype=dtype)
    cx = W * 0.5; cy = H * 0.5

    # ----- W2C, camera-frame points Xc (B,N,3)
    W2C = torch.stack([c.world_view_transform for c in cameras], dim=0).to(dev, dtype)  # (B,4,4)
    Rc = W2C[:, :3, :3]                                         # (B,3,3)

    ones = torch.ones((N, 1), device=dev, dtype=dtype)
    Xw_h = torch.cat([means3d.to(dev, dtype), ones], dim=1)      # (N,4)
    # (B,4,4) @ (4,N) → (B,4,N) → (B,N,3)
    Xc = (W2C @ Xw_h.t().unsqueeze(0)).transpose(-2, -1)[..., :3]  # (B,N,3)
    X, Y, Z = Xc.unbind(-1)                                      # each (B,N)
    invZ  = 1.0 / (Z + 1e-8)
    invZ2 = invZ * invZ

    # centers (OpenCV/COLMAP v-down)
    uc = fx.unsqueeze(1) * (X * invZ) + cx                       # (B,N)
    vc = fy.unsqueeze(1) * (Y * invZ) + cy                       # (B,N)

    # ----- camera-frame Gaussian covariance: Σ_c = R_c Σ_w R_c^T (broadcast over B)
    # Expand Σ_w over B and transform with Rc:
    Sw_b = Sw.unsqueeze(0).expand(B, N, 3, 3)                     # (B,N,3,3)
    Rc_b = Rc.unsqueeze(1).expand(B, N, 3, 3)                     # (B,N,3,3)
    Sc = Rc_b @ Sw_b @ Rc_b.transpose(2, 3)                       # (B,N,3,3)

    # ----- clamped Jacobian J (B,N,2,3); same clamp used in your single-view path
    tx = torch.tan(fovx * 0.5) * clamp_factor                     # (B,)
    ty = torch.tan(fovy * 0.5) * clamp_factor
    xoz = torch.clamp(X * invZ, -tx.unsqueeze(1), tx.unsqueeze(1))
    yoz = torch.clamp(Y * invZ, -ty.unsqueeze(1), ty.unsqueeze(1))

    zeros = torch.zeros_like(invZ)
    J = torch.stack([
        torch.stack([fx.unsqueeze(1) * invZ,          zeros,                   -fx.unsqueeze(1) * xoz * invZ], dim=-1),
        torch.stack([zeros,                           fy.unsqueeze(1) * invZ,  -fy.unsqueeze(1) * yoz * invZ], dim=-1),
    ], dim=-2)                                                    # (B,N,2,3)

    # Σ_2d = J Σ_c J^T  (reshape to (B*N,...) for matmuls, then back)
    J_rs  = J.reshape(B * N, 2, 3)
    Sc_rs = Sc.reshape(B * N, 3, 3)
    S2d   = (J_rs @ Sc_rs @ J_rs.transpose(1, 2)).reshape(B, N, 2, 2)

    # ----- inverse of 2x2 SPD (closed-form)
    a = S2d[..., 0, 0]; b = S2d[..., 0, 1]; c = S2d[..., 1, 1]
    det = torch.clamp(a * c - b * b, min=1e-12)
    Sinv = torch.zeros_like(S2d)
    Sinv[..., 0, 0] =  c / det
    Sinv[..., 0, 1] = -b / det
    Sinv[..., 1, 0] = -b / det
    Sinv[..., 1, 1] =  a / det

    out = {"Sinv": Sinv, "uc": uc, "vc": vc, "Z": Z}

    if return_sigmas:
        # diag(Σ_2d) in pixels → useful for bbox sizing in single-view callers
        eps = torch.tensor(1e-8, device=dev, dtype=dtype)
        sx = torch.sqrt(torch.clamp(S2d[..., 0, 0], min=eps))
        sy = torch.sqrt(torch.clamp(S2d[..., 1, 1], min=eps))
        if max_sigma_px is None:
            max_sigma_px = float(max(W, H) * 2.0)
        sx = torch.clamp(sx, min_sigma_px, max_sigma_px)
        sy = torch.clamp(sy, min_sigma_px, max_sigma_px)
        out["sigmas"] = torch.stack([sx, sy], dim=-1)            # (B,N,2)

    return out

def projected_covariance_2d(
    *,
    means3d: torch.Tensor,        # (N,3)
    scales: torch.Tensor,         # (N,3)  (these are GraphDECO log-scales)
    rotations: torch.Tensor,      # (N,4)  WXYZ
    camera: "Camera",
    min_sigma_px: float = 0.5,
    max_sigma_px: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build 2D screen-space covariance for each Gaussian:
      Sigma_w = R(q) diag(s^2) R(q)^T
      Sigma_c = R_cw Sigma_w R_cw^T
      J = d(u,v)/d(X,Y,Z) at mean
      Sigma_2d = J Sigma_c J^T
    Returns uc, vc, Z, Sinv (2x2 inverse), sigmas (sqrt diag) for bbox.
    """
    out = projected_covariance_2d_batched(
        means3d=means3d,
        scales_log=scales,
        rotations_wxyz=rotations,
        cameras=[camera],
        return_sigmas=True,
        min_sigma_px=min_sigma_px,
        max_sigma_px=max_sigma_px,
        is_render=False,
    )
    # squeeze B=1 back to (N,*) outputs
    squeezed = {
        "Sinv": out["Sinv"].squeeze(0),
        "uc": out["uc"].squeeze(0),
        "vc": out["vc"].squeeze(0),
        "Z": out["Z"].squeeze(0),
        "sigmas": out["sigmas"].squeeze(0),
    }
    return squeezed


# WXYZ quaternion to rotation matrix (matches GraphDECO rot_0..3)
def quat_wxyz_to_rotmat(q: torch.Tensor, is_render = True) -> torch.Tensor:
    """
    q: (N,4) quaternion as (w, x, y, z)  -> (N,3,3) rotation matrices
    This matches GraphDECO's PLY (rot_0..3 is WXYZ).
    """
    if is_render:
        q = torch.nn.functional.normalize(q, dim=-1)
    else:
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    r00 = 1 - 2*(yy + zz); r01 = 2*(xy - wz);   r02 = 2*(xz + wy)
    r10 = 2*(xy + wz);     r11 = 1 - 2*(xx+zz); r12 = 2*(yz - wx)
    r20 = 2*(xz - wy);     r21 = 2*(yz + wx);   r22 = 1 - 2*(xx+yy)
    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).view(-1, 3, 3)
    return R

# ---------- GraphDECO-compatible camera parsing (no file I/O) ----------

def _maybe_rad(a: float) -> float:
    a = float(a)
    # Interpret as radians unless it clearly looks like degrees (> ~3.2 rad)
    return math.radians(a) if a > math.pi * 1.1 else a

def get_dims(fr: Dict[str, Any], defaults: Dict[str, Any]) -> Tuple[int, int]:
    w = fr.get("w") or fr.get("width") or fr.get("W") or fr.get("image_width") or defaults.get("w") or defaults.get("width")
    h = fr.get("h") or fr.get("height") or fr.get("H") or fr.get("image_height") or defaults.get("h") or defaults.get("height")
    if w is None or h is None:
        raise ValueError("Camera JSON missing image size (w/h or width/height).")
    return int(w), int(h)

def fovs_from_intrinsics_dict(fr: Dict[str, Any], w: int, h: int) -> Tuple[float, float]:
    # 1) FoV directly (preferred by GraphDECO dumps)
    for kx, ky in (("FoVx","FoVy"), ("fov_x","fov_y"), ("fovX","fovY")):
        if kx in fr and ky in fr:
            return _maybe_rad(fr[kx]), _maybe_rad(fr[ky])

    # 2) focal pixels
    for kfx, kfy in (("fl_x","fl_y"), ("fx","fy")):
        if kfx in fr and kfy in fr:
            fx = float(fr[kfx]); fy = float(fr[kfy])
            fovx = 2.0 * math.atan((w * 0.5) / max(fx, 1e-8))
            fovy = 2.0 * math.atan((h * 0.5) / max(fy, 1e-8))
            return fovx, fovy

    # 3) intrinsics matrix K
    K = fr.get("K") or fr.get("k") or (fr.get("intrinsics") or {}).get("K")
    if K is not None:
        K = np.array(K, dtype=np.float32).reshape(3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        fovx = 2.0 * math.atan((w * 0.5) / max(fx, 1e-8))
        fovy = 2.0 * math.atan((h * 0.5) / max(fy, 1e-8))
        return fovx, fovy

    # 4) projection matrix → FoV (OpenGL-style)
    for key in ("projection_matrix", "projection", "proj_matrix", "P"):
        if key in fr:
            P = as_4x4_np(fr[key])
            fovx = 2.0 * math.atan(1.0 / max(P[0, 0], 1e-8))
            fovy = 2.0 * math.atan(1.0 / max(P[1, 1], 1e-8))
            return fovx, fovy

    raise ValueError("Need FoV (FoVx/FoVy), focal pixels (fx/fy or fl_x/fl_y), K, or a projection matrix.")

def w2c_from_dict(fr: Dict[str, Any]) -> np.ndarray:
    # Direct world->camera 4x4
    for key in ("world_view_transform", "world_view", "view_matrix", "world_to_camera", "w2c"):
        if key in fr:
            return as_4x4_np(fr[key]).astype(np.float32)

    # camera->world 4x4 (invert)
    for key in ("camera_to_world", "c2w", "transform_matrix", "pose", "transform", "matrix", "extrinsic_matrix"):
        if key in fr:
            return np.linalg.inv(as_4x4_np(fr[key])).astype(np.float32)

    # R/t (assumed world->camera) as fallback
    R = fr.get("R") or fr.get("rotation") or (fr.get("extrinsics") or {}).get("R")
    t = fr.get("T") or fr.get("t") or fr.get("translation") or (fr.get("extrinsics") or {}).get("t")
    if R is not None and t is not None:
        R = np.array(R, dtype=np.float32).reshape(3, 3)
        t = np.array(t, dtype=np.float32).reshape(3)
        M = np.eye(4, dtype=np.float32); M[:3, :3] = R; M[:3, 3] = t
        return M

    # Quaternion + position (assume camera_to_world, invert)
    q = fr.get("quaternion") or fr.get("q") or (fr.get("extrinsics") or {}).get("quaternion")
    p = fr.get("position") or fr.get("C") or fr.get("cam_center") or (fr.get("extrinsics") or {}).get("position")
    if q is not None and p is not None:
        # try XYZW and WXYZ interpretations
        def _quat_to_R_np(qv: np.ndarray, order: str) -> np.ndarray:
            qv = np.array(qv, dtype=np.float32).reshape(4)
            if order == "wxyz": w, x, y, z = qv[0], qv[1], qv[2], qv[3]
            else:               x, y, z, w = qv[0], qv[1], qv[2], qv[3]
            n = math.sqrt(max(1e-12, x*x + y*y + z*z + w*w))
            w, x, y, z = w/n, x/n, y/n, z/n
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            return np.array([[1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
                             [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
                             [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]], dtype=np.float32)
        for order in ("xyzw", "wxyz"):
            R = _quat_to_R_np(q, order=order)
            c2w = np.eye(4, dtype=np.float32); c2w[:3, :3] = R; c2w[:3, 3] = np.array(p, dtype=np.float32).reshape(3)
            w2c = np.linalg.inv(c2w).astype(np.float32)
            if np.isfinite(w2c).all():
                return w2c

    # COLMAP-style qvec (w,x,y,z) + tvec (already world->camera)
    qvec = fr.get("qvec") or fr.get("q_wxyz")
    tvec = fr.get("tvec")
    if qvec is not None and tvec is not None:
        w, x, y, z = np.array(qvec, dtype=np.float32).reshape(4)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R = np.array([[1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
                      [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
                      [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)]], dtype=np.float32)
        M = np.eye(4, dtype=np.float32); M[:3, :3] = R; M[:3, 3] = np.array(tvec, dtype=np.float32).reshape(3)
        return M

    # rotation (3x3) + position (3,) as camera_to_world (invert)
    R = fr.get("rotation"); p = fr.get("position")
    if R is not None and p is not None:
        c2w = np.eye(4, dtype=np.float32); c2w[:3, :3] = np.array(R, dtype=np.float32).reshape(3, 3)
        c2w[:3, 3] = np.array(p, dtype=np.float32).reshape(3)
        return np.linalg.inv(c2w).astype(np.float32)

    raise ValueError("No extrinsics in frame (need world_view/world_to_camera/w2c, or camera_to_world/c2w, or R/t, or quaternion+position).")


def _camera_from_frame_cpu(fr: Dict[str, Any], defaults: Dict[str, Any]) -> Camera:
    device = torch.device("cpu")
    w, h = get_dims(fr, defaults)
    fovx, fovy = fovs_from_intrinsics_dict(fr, w, h)
    w2c = w2c_from_dict(fr)
    w2c_t = torch.tensor(w2c, dtype=torch_ctx.dtype, device=device)
    cam = Camera.from_view_fov(w2c_t, w, h, fovx, fovy)
    try:
        c2w = torch.linalg.inv(w2c_t)
        cam.camera_center = c2w[:3, 3]
    except Exception:
        cam.camera_center = None
    return cam

def camera_from_frame(fr: Dict[str, Any], defaults: Dict[str, Any], device: Optional[torch.device] = None) -> Camera:
    """
    Wrapper of _camera_from_frame_cpu for backwards compatibility.
    Ignores device, always uses CPU.
    """
    return _camera_from_frame_cpu(fr, defaults)