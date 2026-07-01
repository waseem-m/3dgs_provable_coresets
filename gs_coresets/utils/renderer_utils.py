# renderer_utils_vectorized_batch.py
# Batched, GraphDECO-compatible renderer (multiple cameras, no custom CUDA)
# Assumes: all cameras share W,H and tile size (BLOCK_X/BLOCK_Y)
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from gs_coresets.utils.camera_utils import Camera, projected_covariance_2d_batched
from gs_coresets.utils.gaussian_utils import GaussianPLY
from gs_coresets.utils.io_utils import ensure_dir_for
from gs_coresets.utils.types_devices_utils import torch_ctx

# --- SH constants (match 3DGS CUDA) ---
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396]
SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435]


@torch.inference_mode()
def compute_color_from_sh(
    features_dc: torch.Tensor,      # (N,3,1)
    features_rest: torch.Tensor,    # (N,3,M-1)
    viewdirs: torch.Tensor,         # (N,3) unit directions (from camera to Gaussian)
    deg: int = 3
) -> torch.Tensor:
    """
    Matches GraphDECO CUDA path: channel-major coeff ordering, +0.5 bias and clamp to >= 0.
    Returns (N,3) in linear space.
    """
    if deg < 0 or deg > 3:
        raise ValueError("Supported SH degrees: 0..3")

    N = features_dc.shape[0]
    M = (deg + 1) ** 2
    assert features_dc.shape == (N, 3, 1)
    assert features_rest.shape == (N, 3, M - 1)
    assert viewdirs.shape == (N, 3)

    x = viewdirs[..., 0:1]
    y = viewdirs[..., 1:2]
    z = viewdirs[..., 2:3]

    B0 = SH_C0 * torch.ones_like(x)  # (N,1)

    if deg == 0:
        B = B0
    else:
        B1 = torch.cat([-SH_C1 * y, SH_C1 * z, -SH_C1 * x], dim=-1)  # (N,3)
        if deg == 1:
            B = torch.cat([B0, B1], dim=-1)
        else:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            B2 = torch.cat([
                SH_C2[0] * xy,
                SH_C2[1] * yz,
                SH_C2[2] * (2.0 * zz - xx - yy),
                SH_C2[3] * xz,
                SH_C2[4] * (xx - yy),
            ], dim=-1)  # (N,5)
            if deg == 2:
                B = torch.cat([B0, B1, B2], dim=-1)
            else:
                B3 = torch.cat([
                    SH_C3[0] * y * (3 * x * x - y * y),
                    SH_C3[1] * x * y * z,
                    SH_C3[2] * y * (4 * z * z - x * x - y * y),
                    SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y),
                    SH_C3[4] * x * (4 * z * z - x * x - y * y),
                    SH_C3[5] * z * (x * x - y * y),
                    SH_C3[6] * x * (x * x - 3 * y * y),
                ], dim=-1)  # (N,7)
                B = torch.cat([B0, B1, B2, B3], dim=-1)  # (N,16) for deg=3

    dc_term = (features_dc.squeeze(-1) * B[:, :1]).squeeze(-1)  # (N,3)
    rest_term = torch.einsum("ncm,nm->nc", features_rest, B[:, 1:])
    color = dc_term + rest_term + 0.5
    color = torch.clamp(color, min=0.0)
    return color

# bit masks
PER_CHANNEL = 1 << 0
PER_PIXEL   = 1 << 1
PER_TILE    = 1 << 2
PER_IMAGE   = 1 << 3
PER_BATCH   = 1 << 4
PER_SCENE   = 1 << 5

class BatchedRenderer:
    """
    Render multiple cameras in one call (no custom CUDA).
    Identical per-view output to your single-camera vectorized renderer.
    """
    @torch.inference_mode()
    def __init__(self, cameras: List[Camera], bg=(0.0, 0.0, 0.0), device=torch_ctx.device, white_background=False, block_xy: int = 16):
        assert len(cameras) >= 1, "Need at least one camera"
        self.cameras = cameras
        self.B = len(cameras)
        self.image_W = cameras[0].width
        self.image_H = cameras[0].height
        assert all(cam.width == self.image_W and cam.height == self.image_H for cam in cameras
                   ), "All cameras must share W,H"
        self.bg = torch.tensor(bg, dtype=torch_ctx.dtype)
        self.white_background = white_background
        self.device = device
        self.BLOCK_X = block_xy
        self.BLOCK_Y = block_xy

    @torch.inference_mode()
    def render_batch(self,
                     model_ply: GaussianPLY,
                     per_channel_sensitivity: bool = True,
                     per_pixel_sensitivity: bool = True,
                     per_tile_sensitivity: bool = True,
                     per_image_sensitivity: bool = True,
                     per_batch_sensitivity: bool = True,
                     per_scene_sensitivity: bool = True,
                     sensitivity_norm: str = "l1",
                     sensitivity_reduce: str = "max",
                     sensitivity_nocolor: bool = False) -> dict:
        if sensitivity_norm not in {"l1", "l2"}:
            raise ValueError(f"Unsupported sensitivity_norm '{sensitivity_norm}' (expected 'l1' or 'l2')")
        if sensitivity_reduce not in {"max", "mean"}:
            raise ValueError(f"Unsupported sensitivity_reduce '{sensitivity_reduce}' (expected 'max' or 'mean')")
        reduce_is_mean = (sensitivity_reduce == "mean")
        dev = self.device
        B, H, W = self.B, self.image_H, self.image_W
        cams = self.cameras

        # Sensitivity flags
        sensitivity_flags = (
                (int(per_channel_sensitivity) << 0) |
                (int(per_pixel_sensitivity) << 1) |
                (int(per_tile_sensitivity) << 2) |
                (int(per_image_sensitivity) << 3) |
                (int(per_batch_sensitivity) << 4) |
                (int(per_scene_sensitivity) << 5)
        )
        sens_need_per_channel = bool(sensitivity_flags >= PER_CHANNEL)
        sens_need_per_pixel = bool(sensitivity_flags >= PER_PIXEL)
        sens_need_per_tile = bool(sensitivity_flags >= PER_TILE)
        sens_need_per_image = bool(sensitivity_flags >= PER_IMAGE)
        sens_need_per_batch= bool(sensitivity_flags >= PER_BATCH)
        sens_need_per_scene = bool(sensitivity_flags >= PER_SCENE)

        #use_amp = (dev.type == "cuda")
        use_amp = False
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            # Inputs
            means = model_ply.means.to(dev, torch_ctx.dtype)                 # (N,3)
            scales = model_ply.scales.to(dev, torch_ctx.dtype)               # (N,3) log
            rots_wxyz = model_ply.rotations_wxyz.to(dev, torch_ctx.dtype)    # (N,4)
            opacity_logits = model_ply.opacity_logits.to(dev, torch_ctx.dtype).view(-1)  # (N,)
            fdc = model_ply.features_dc.to(dev, torch_ctx.dtype)             # (N,3,1)
            frest = model_ply.features_rest.to(dev, torch_ctx.dtype)         # (N,3,M-1)
            alpha_vec = torch.sigmoid(opacity_logits)                # (N,)
            # N = number of Gaussians
            N = means.shape[0]
            use_l2 = (sensitivity_norm == "l2")

            # Σ2D^-1, (u,v), Z for all cameras
            cov = projected_covariance_2d_batched(
                means3d=means,
                scales_log=scales,  # this should be the GraphDECO *log* scales
                rotations_wxyz=rots_wxyz,  # (N,4) WXYZ
                cameras=cams,
                clamp_factor=1.3,
            )
            Sinv = cov["Sinv"]              # (B,N,2,2)
            uc = cov["uc"]; vc = cov["vc"]; Z = cov["Z"]  # (B,N)

            # Colors per camera (viewdirs in CAMERA space → robust)
            # Xc already computed in projected_covariance_2d_batched; recompute cheaply:
            W2C = torch.stack([c.world_view_transform for c in cams], dim=0).to(dev, torch_ctx.dtype)  # (B,4,4)
            ones = torch.ones((N,1), device=dev, dtype=torch_ctx.dtype)
            Xw_h = torch.cat([means, ones], dim=1)  # (N,4)
            Xc = (W2C @ Xw_h.t().unsqueeze(0)).transpose(-2, -1)[..., :3] # (B,N,3)
            viewdirs = torch.nn.functional.normalize(Xc, dim=-1)     # (B,N,3)
            # compute SH in batch → (B,N,3)
            M_tot = fdc.shape[-1] + frest.shape[-1]
            deg = 3 if M_tot >= 16 else (2 if M_tot >= 9 else (1 if M_tot >= 4 else 0))
            # vectorize compute_color_from_sh over B
            rgb_all = []
            for b in range(B):
                rgb_b = compute_color_from_sh(fdc, frest, viewdirs[b], deg=deg)  # (N,3)
                rgb_all.append(rgb_b)
            rgb_all = torch.stack(rgb_all, dim=0)  # (B,N,3)

            # Output tensors
            out_color = torch.zeros((B, H, W, 3), dtype=torch_ctx.dtype, device=dev)
            out_depth = torch.zeros((B, H, W), dtype=torch_ctx.dtype, device=dev)
            trans     = torch.ones((B, H, W), dtype=torch_ctx.dtype, device=dev)

            # Background
            bg = self.bg.to(dev, torch_ctx.dtype)
            if self.white_background:
                bg = torch.ones_like(bg)

            # Tiles
            tilesize_X, tilesize_Y = self.BLOCK_X, self.BLOCK_Y
            grid_x = (W + tilesize_X - 1) // tilesize_X
            grid_y = (H + tilesize_Y - 1) // tilesize_Y
            # Number of tiles per camera/image
            num_tiles_per_cam = grid_x * grid_y

            # --- radius from Σ2D eigenvalues (B,N) + 1px guard ---
            a = Sinv[..., 0, 0]; b = Sinv[..., 0, 1]; c = Sinv[..., 1, 1]
            det = a*c - b*b
            det = torch.clamp(det, min=1e-12)
            S_xx = c / det; S_xy = -b / det; S_yy = a / det
            mid = 0.5 * (S_xx + S_yy)
            disc = torch.clamp(mid*mid - (S_xx*S_yy - S_xy*S_xy), min=0.0)
            lam1 = mid + torch.sqrt(disc)
            lam2 = mid - torch.sqrt(disc)
            lam_max = torch.maximum(lam1, lam2)               # (B,N)
            my_radius = torch.ceil(3.0 * torch.sqrt(lam_max) + 1.0)  # (B,N)

            # --- Visibility + vectorized binning across all cameras ---
            visible = Z > 0.2                                   # (B,N)
            # tile rects
            rx0 = torch.clamp(((uc - my_radius) / tilesize_X).floor().to(torch.int32), 0, grid_x)
            ry0 = torch.clamp(((vc - my_radius) / tilesize_Y).floor().to(torch.int32), 0, grid_y)
            rx1 = torch.clamp(((uc + my_radius) / tilesize_X).floor().to(torch.int32) + 1, 0, grid_x)
            ry1 = torch.clamp(((vc + my_radius) / tilesize_Y).floor().to(torch.int32) + 1, 0, grid_y)
            has_area = (rx1 > rx0) & (ry1 > ry0)
            vis = visible & has_area

            # indices of (cam, gaussian) that are visible
            pairs = torch.nonzero(vis, as_tuple=False)  # (K,2) [b, n]
            if pairs.numel() == 0:
                zeros_N = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                zero_scalar = torch.tensor(0.0, device=dev, dtype=torch_ctx.dtype)
                out = {
                    "rgb": out_color.clamp(0.0, 1.0),
                    "alpha": (1.0 - trans).clamp(0.0, 1.0),
                    "depth": out_depth
                }
                if per_channel_sensitivity:
                    out["sens_channel"] = zeros_N.clamp(0.0, 1.0)
                if per_pixel_sensitivity:
                    out["sens_pixel"] = zeros_N.clamp(0.0, 1.0)
                if per_tile_sensitivity:
                    out["sens_tile"] = zeros_N.clamp(0.0, 1.0)
                if per_image_sensitivity:
                    out["sens_image"] = zeros_N.clamp(0.0, 1.0)
                if per_batch_sensitivity:
                    out["sens_batch"] = zeros_N.clamp(0.0, 1.0)
                if per_scene_sensitivity:
                    out["nom_batch"] = zeros_N
                    out["denom_batch_raw"] = zero_scalar
                return out

            # Sensitivity tensors initializations:
            gauss_batch_sens_max_ch = None
            gauss_batch_sens_sum_ch = None
            gauss_batch_sens_count_ch = None
            gauss_batch_sens_max_pix = None
            gauss_batch_sens_sum_pix = None
            gauss_batch_sens_count_pix = None
            gauss_batch_sens_max_tile = None
            gauss_batch_sens_sum_tile = None
            gauss_batch_sens_count_tile = None
            gauss_contrib_curr_img = None
            denom_curr_img_raw = None
            if per_channel_sensitivity:
                if reduce_is_mean:
                    gauss_batch_sens_sum_ch = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                    gauss_batch_sens_count_ch = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                else:
                    gauss_batch_sens_max_ch = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
            if per_pixel_sensitivity:
                if reduce_is_mean:
                    gauss_batch_sens_sum_pix = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                    gauss_batch_sens_count_pix = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                else:
                    gauss_batch_sens_max_pix = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
            if per_tile_sensitivity:
                if reduce_is_mean:
                    gauss_batch_sens_sum_tile = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                    gauss_batch_sens_count_tile = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                else:
                    gauss_batch_sens_max_tile = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
            if sens_need_per_image:
                cur_cam = None
                gauss_contrib_curr_img = torch.zeros(N, device=dev, dtype=torch_ctx.dtype)
                denom_curr_img_raw = torch.tensor(0.0, device=dev, dtype=torch_ctx.dtype)
            if per_image_sensitivity:
                if reduce_is_mean:
                    gauss_batch_sens_sum_img = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                    gauss_batch_sens_count_img = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                else:
                    gauss_batch_sens_max_img = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
            if sens_need_per_batch:
                gauss_contrib_per_batch = torch.zeros(N, dtype=torch_ctx.dtype, device=dev)
                denom_per_batch_raw = torch.tensor(0.0, device=dev, dtype=torch_ctx.dtype)

            b_id = pairs[:, 0]  # (K,)
            n_id = pairs[:, 1]  # (K,)
            r0x = rx0[b_id, n_id]; r0y = ry0[b_id, n_id]
            r1x = rx1[b_id, n_id]; r1y = ry1[b_id, n_id]
            nx = (r1x - r0x); ny = (r1y - r0y)
            counts = nx * ny
            keep = counts > 0
            if not torch.all(keep):
                b_id = b_id[keep]; n_id = n_id[keep]
                r0x = r0x[keep]; r0y = r0y[keep]; nx = nx[keep]; ny = ny[keep]
            K = b_id.numel()

            # chunked expansion to (tile_global, gaussian_id)
            tile_ids_all = []
            gidx_all     = []
            b_all        = []
            CHUNK = 32768
            start = 0
            while start < K:
                end = min(start + CHUNK, K)
                bx = b_id[start:end]; nn = n_id[start:end]
                _r0x = r0x[start:end]; _r0y = r0y[start:end]
                _nx  = nx[start:end];  _ny  = ny[start:end]
                Gc = bx.shape[0]
                if Gc == 0:
                    start = end; continue
                max_nx = int(_nx.max().item()); max_ny = int(_ny.max().item())
                if max_nx == 0 or max_ny == 0:
                    start = end; continue
                jx = torch.arange(max_nx, device=dev)
                iy = torch.arange(max_ny, device=dev)
                gx = _r0x.view(Gc,1,1) + jx.view(1,1,max_nx)
                gy = _r0y.view(Gc,1,1) + iy.view(1,max_ny,1)
                gx = gx.expand(-1, max_ny, -1)
                gy = gy.expand(-1, -1, max_nx)
                mask_x = jx.view(1,1,max_nx) < _nx.view(Gc,1,1)
                mask_y = iy.view(1,max_ny,1) < _ny.view(Gc,1,1)
                m = mask_x & mask_y
                tile_lin = (gy * grid_x + gx)[m]  # (Kc,)
                tile_global = bx.view(Gc,1,1).expand_as(gx)[m] * num_tiles_per_cam + tile_lin
                tile_ids_all.append(tile_global.to(torch.long))
                gidx_all.append(nn.view(Gc,1,1).expand_as(gx)[m].to(torch.long))
                b_all.append(bx.view(Gc,1,1).expand_as(gx)[m].to(torch.long))
                start = end

            tile_ids = torch.cat(tile_ids_all, dim=0)  # (M,)
            gidx = torch.cat(gidx_all, dim=0)          # (M,)
            bidx = torch.cat(b_all, dim=0)             # (M,)

            # group by tile_global
            order = torch.argsort(tile_ids)
            tile_ids = tile_ids[order]
            gidx = gidx[order]; bidx = bidx[order]

            total_tiles = B * num_tiles_per_cam
            tile_counts  = torch.bincount(tile_ids, minlength=total_tiles)  # (Ttot,)
            tile_offsets = torch.cumsum(tile_counts, dim=0)
            tile_starts  = torch.empty_like(tile_offsets)
            tile_starts[0] = 0
            tile_starts[1:] = tile_offsets[:-1]

            # render per global tile (cam,ty,tx); per-tile kernel stays vectorized
            for tile_gl in range(total_tiles):
                start = tile_starts[tile_gl].item()
                end   = tile_offsets[tile_gl].item()
                if start == end:
                    continue

                cam_id = tile_gl // num_tiles_per_cam
                tloc   = tile_gl % num_tiles_per_cam
                ty = tloc // grid_x
                tx = tloc %  grid_x
                y_min = ty * tilesize_Y; y_max = min(y_min + tilesize_Y, H)
                x_min = tx * tilesize_X; x_max = min(x_min + tilesize_X, W)

                idxs_t = gidx[start:end]  # (G,)
                # depth sort (near→far) for this camera
                zloc = Z[cam_id, idxs_t]
                ord2 = torch.argsort(zloc)
                idxs_t = idxs_t[ord2]

                # gather per-gaussian, per-camera
                u = uc[cam_id, idxs_t]
                v = vc[cam_id, idxs_t]
                S = Sinv[cam_id, idxs_t]  # (G,2,2)
                A_t = S[:,0,0]; B_t = S[:,0,1]; C_t = S[:,1,1]
                rgb = rgb_all[cam_id, idxs_t]        # (G,3)
                alpha_g = alpha_vec[idxs_t]          # (G,)
                zvals = Z[cam_id, idxs_t]            # (G,)

                # tile pixel grid
                ys = torch.arange(y_min, y_max, device=dev)
                xs = torch.arange(x_min, x_max, device=dev)
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                pix_x = gx.reshape(-1).to(torch_ctx.dtype)  # (P,)
                pix_y = gy.reshape(-1).to(torch_ctx.dtype)
                P = pix_x.numel()
                G = idxs_t.shape[0]
                if G == 0 or P == 0:
                    continue

                # vectorized tile kernel (identical math to single-view)
                dx = u.view(G,1) - pix_x.view(1,P)
                dy = v.view(G,1) - pix_y.view(1,P)
                power_all = -0.5 * (A_t.view(G,1)*dx*dx + 2.0*B_t.view(G,1)*dx*dy + C_t.view(G,1)*dy*dy)

                conic = -2.0 * power_all
                valid_power_conic = (conic <= 9.0)

                alpha_eps = 1.0/255.0
                alpha_g_safe = torch.clamp(alpha_g, min=1e-8)
                thresh = (math.log(alpha_eps) - torch.log(alpha_g_safe)).view(G,1)
                valid_power_eps = (power_all >= thresh)

                valid_power = valid_power_conic & valid_power_eps
                power_masked = torch.where(valid_power, power_all, torch.full_like(power_all, float("-inf")))

                # alpha_px = actual per-(g,p) opacity (alpha_i*gauss_i) used in composition.
                alpha_px = alpha_g.view(G,1) * torch.exp(power_masked)  # (G,P)
                alpha_px = torch.minimum(torch.tensor(0.99, device=dev, dtype=torch_ctx.dtype), alpha_px)

                # one_minus = Per-(g,p) transparency factor.
                one_minus = 1.0 - alpha_px

                # T_prefix[i] = T_i = Mul(1 - alpha_px[i]) = Mul(one_minus[i])
                T_prefix = torch.cumprod(
                    torch.cat([torch.ones((1,P), device=dev, dtype=torch_ctx.dtype),
                               one_minus[:-1,:]], dim=0),
                    dim=0
                )
                # Early termination mask = 1e-4
                active_prefix = (T_prefix >= 1e-4)

                # Per-(g,p) compositing weight, zeroed out once the pixel is “opaque enough” by the threshold.
                wgt = (T_prefix * alpha_px) * active_prefix.to(alpha_px.dtype)  # (G,P)

                ##  i is a Gaussian.
                ##  p is a pixel.
                ##  wgt[i,p] = alpha[i]*gauss[i,p]*T[i,p] ==> "Gaussian's intensity"
                ##  rgb[i,r/g/b]
                ##  Ctile[p,r/g/b] = sum_i(wgt[i,p] * rgb[i,r/g/b])
                # "gp,gc->pc":
                # multiply along the shared g index and sum over g (Gaussians),
                # producing an output indexed by p (pixels) and c (channels).
                # identical, just more explicit:
                # Ctile = wgt.transpose(0, 1) @ rgb  # (P,G) @ (G,3) -> (P,3)
                Ctile = torch.einsum("gp,gc->pc", wgt, rgb)  # (P,3)

                # Depth map of tile
                Dtile = torch.einsum("gp,g->p",  wgt, zvals)

                # Ttile a very close (and safe) approximation to the exact final transmittance,
                # but it stops once the remaining headroom is negligible - saving compute and preventing underflow.
                one_minus_masked = torch.where(active_prefix, one_minus, torch.ones_like(one_minus))
                Ttile = torch.prod(one_minus_masked, dim=0)   # (P,)

                ####################################################################################################
                #########################################
                ## Intra-tile sensitivity calculations ##
                #########################################
                # N = total number of gaussians in the scene.
                # B = number of cameras (images) in the current batch
                # ----> cam_id: int camera index
                # G = number of Gaussians overlapping this tile
                # ----> idxs_t: global Gaussian indices for this tile -- size=(G,)
                # P = number of pixels in this tile -- (size=tile_h*tile_w)
                # wgt = transmittance-weighted alpha (alpha_i*k_i*T_i) of gaussian g at pixel p -- (size=GxP)
                # rgb = SH-evaluated color of Gaussian g in this camera -- size=(Gx3)
                # Ctile = Current accumulated color of the tile (current sum of contr.) -- size=(P,3)
                # Ttile = Current remaining transmittance (what hasn't been covered) -- size=(P,)
                # bg = background color -- size=(3,)

                # If need to calculate any sensitivity, find each Gaussian's effective contribution (after
                # clamping) to this tile's (pixel,channel) pair.
                if sens_need_per_channel:
                    rgb_sens = torch.ones_like(rgb) if sensitivity_nocolor else rgb
                    contrib_gpc = (wgt.unsqueeze(-1) * rgb_sens.unsqueeze(1))  # (G,P,3)
                    sens_ctile = contrib_gpc.sum(dim=0) if sensitivity_nocolor else Ctile  # (P,3)

                    # Reconstruct per-channel color at this stage (C + T*bg), and clamp denom
                    per_channel_lin_val = sens_ctile + Ttile.unsqueeze(-1) * bg  # (P,3) >= 0
                    denom_per_ch_raw = per_channel_lin_val.clamp(0.0, 1.0) # EXACT val of out["rgb"]

                    # ---- Saturation-aware per-Gaussian contributions (vectorized over channels) ----
                    # contrib[g,p,c] = wgt[g,p] * rgb[g,c]  >= 0

                    # Channel-wise headroom (how much Gaussian sum can still fit before clamp)
                    # cap_pc[p,c] = 1 - Ttile[p] * bg[c]  (clamped at 0)
                    cap_pc = (1.0 - Ttile.unsqueeze(-1) * bg).clamp_min(0.0)  # (P,3)

                    # cumulative prefix over Gaussians (front->back)
                    cum_gpc = contrib_gpc.cumsum(dim=0)  # (G,P,3)
                    prev_gpc = cum_gpc - contrib_gpc  # (G,P,3)

                    # Remaining headroom just before each Gaussian
                    remaining_gpc = (cap_pc.unsqueeze(0) - prev_gpc).clamp_min(0)  # (G,P,3)

                    # Effective (display-relevant) contribution for each Gaussian, pixel, channel
                    eff_gpc = torch.minimum(contrib_gpc, remaining_gpc)  # (G,P,3)

                # ------------------------------------------------------------------
                # A) Gaussian's per-channel sensitivity: max over {Cameras,Camera_Tiles,Pixels,Channels} of
                #      Gaussian's fractional contribution to a channel
                #      --> write into gauss_sens_max_ch[cam_id,gaussians_ids_per_tile]
                #          then take max over batches
                # ------------------------------------------------------------------
                if per_channel_sensitivity:
                    denom_per_ch = denom_per_ch_raw.clamp_min(torch_ctx.denom_eps)  # (P,3)
                    denom_per_ch_sq = denom_per_ch_raw.pow(2).clamp_min(torch_ctx.denom_eps) if use_l2 else None
                    if reduce_is_mean:
                        gauss_tile_sens_sum_ch = torch.zeros((G,), device=dev, dtype=torch_ctx.dtype)
                        gauss_tile_sens_count_ch = torch.zeros((G,), device=dev, dtype=torch_ctx.dtype)
                    else:
                        gauss_tile_sens_max_ch = torch.zeros((G,), device=dev, dtype=torch_ctx.dtype)
                    for ch in range(3):
                        gauss_contrib_this_ch = eff_gpc[..., ch]  # (G,P)
                        if use_l2:
                            gauss_contrib_this_ch = gauss_contrib_this_ch.pow(2)
                            denom_this_ch = denom_per_ch_sq[:, ch]
                        else:
                            denom_this_ch = denom_per_ch[:, ch]
                        gauss_ratio_this_ch = gauss_contrib_this_ch / denom_this_ch.unsqueeze(0)  # (G,P)
                        if reduce_is_mean:
                            gauss_tile_sens_sum_ch += gauss_ratio_this_ch.sum(dim=1)
                            pixel_count = float(gauss_ratio_this_ch.shape[1])
                            gauss_tile_sens_count_ch += torch.full(
                                (G,), pixel_count, dtype=torch_ctx.dtype, device=dev
                            )
                        else:
                            gauss_tile_sens_max_ch = torch.maximum(
                                gauss_tile_sens_max_ch, gauss_ratio_this_ch.max(dim=1).values
                            )  # (G,)
                    if reduce_is_mean:
                        gauss_batch_sens_sum_ch.index_add_(0, idxs_t, gauss_tile_sens_sum_ch)
                        gauss_batch_sens_count_ch.index_add_(0, idxs_t, gauss_tile_sens_count_ch)
                    else:
                        gauss_batch_sens_max_ch[idxs_t] = torch.maximum(
                            gauss_batch_sens_max_ch[idxs_t], gauss_tile_sens_max_ch
                        )  # (N,)

                gauss_contrib_per_pix_norm = None
                denom_per_pix_norm = None
                if sens_need_per_pixel:
                    gauss_contrib_per_pix = eff_gpc.sum(dim=2)  # (G,P)
                    denom_per_pix_raw = denom_per_ch_raw.sum(dim=1)  # (P,)
                    if use_l2:
                        gauss_contrib_per_pix_norm = gauss_contrib_per_pix.pow(2)
                        denom_per_pix_norm = denom_per_pix_raw.pow(2)
                    else:
                        gauss_contrib_per_pix_norm = gauss_contrib_per_pix
                        denom_per_pix_norm = denom_per_pix_raw

                if per_pixel_sensitivity:
                    denom_per_pix = denom_per_pix_norm.clamp_min(torch_ctx.denom_eps)
                    gauss_ratio_per_pix = gauss_contrib_per_pix_norm / denom_per_pix.unsqueeze(0)  # (G,P)
                    if reduce_is_mean:
                        gauss_tile_sens_sum_pix = gauss_ratio_per_pix.sum(dim=1)
                        pixel_count = float(gauss_ratio_per_pix.shape[1])
                        gauss_tile_sens_count_pix = torch.full(
                            (G,), pixel_count, dtype=torch_ctx.dtype, device=dev
                        )
                        gauss_batch_sens_sum_pix.index_add_(0, idxs_t, gauss_tile_sens_sum_pix)
                        gauss_batch_sens_count_pix.index_add_(0, idxs_t, gauss_tile_sens_count_pix)
                    else:
                        gauss_tile_sens_max_pix = gauss_ratio_per_pix.max(dim=1).values  # (G,)
                        gauss_batch_sens_max_pix[idxs_t] = torch.maximum(
                            gauss_batch_sens_max_pix[idxs_t], gauss_tile_sens_max_pix
                        )  # (N,)

                gauss_contrib_per_tile = None
                denom_per_tile_raw = None
                if sens_need_per_tile:
                    gauss_contrib_per_tile = gauss_contrib_per_pix_norm.sum(dim=1)  # (G,)
                    denom_per_tile_raw = denom_per_pix_norm.sum()  # ()

                if per_tile_sensitivity:
                    denom_per_tile = denom_per_tile_raw.clamp_min(torch_ctx.denom_eps)
                    gauss_ratio_per_tile = gauss_contrib_per_tile / denom_per_tile  # (G,)
                    if reduce_is_mean:
                        gauss_tile_sens_count_tile = torch.ones((G,), dtype=torch_ctx.dtype, device=dev)
                        gauss_batch_sens_sum_tile.index_add_(0, idxs_t, gauss_ratio_per_tile)
                        gauss_batch_sens_count_tile.index_add_(0, idxs_t, gauss_tile_sens_count_tile)
                    else:
                        gauss_batch_sens_max_tile[idxs_t] = torch.maximum(
                            gauss_batch_sens_max_tile[idxs_t], gauss_ratio_per_tile
                        )  # (N,)

                if sens_need_per_image:
                    if gauss_contrib_per_tile is None or denom_per_tile_raw is None:
                        raise RuntimeError("per-image sensitivity requested without per-tile accumulators")
                    if cur_cam is None:
                        cur_cam = cam_id
                    elif cam_id != cur_cam:
                        if per_image_sensitivity:
                            denom_per_img = denom_curr_img_raw.clamp_min(torch_ctx.denom_eps)
                            gauss_batch_ratio_curr_img = gauss_contrib_curr_img / denom_per_img  # (N,)
                            if reduce_is_mean:
                                contrib_mask = (gauss_contrib_curr_img > 0).to(torch_ctx.dtype)
                                gauss_batch_sens_sum_img += gauss_batch_ratio_curr_img
                                gauss_batch_sens_count_img += contrib_mask
                            else:
                                gauss_batch_sens_max_img = torch.maximum(
                                    gauss_batch_sens_max_img, gauss_batch_ratio_curr_img
                                )
                        if sens_need_per_batch:
                            gauss_contrib_per_batch += gauss_contrib_curr_img
                            denom_per_batch_raw += denom_curr_img_raw
                        gauss_contrib_curr_img.zero_()
                        denom_curr_img_raw.zero_()
                        cur_cam = cam_id

                    gauss_contrib_curr_img.index_add_(0, idxs_t, gauss_contrib_per_tile)
                    denom_curr_img_raw += denom_per_tile_raw

                Ctile_img = Ctile.view(y_max - y_min, x_max - x_min, 3)
                Ttile_img = Ttile.view(y_max - y_min, x_max - x_min)
                Dtile_img = Dtile.view(y_max - y_min, x_max - x_min)

                out_color[cam_id, y_min:y_max, x_min:x_max, :] = Ctile_img + Ttile_img.unsqueeze(-1) * bg
                out_depth[cam_id, y_min:y_max, x_min:x_max] = Dtile_img
                trans[cam_id, y_min:y_max, x_min:x_max] = Ttile_img

            out = {
                "rgb": out_color.clamp(0.0, 1.0),  # (B,H,W,3)
                "alpha": (1.0 - trans).clamp(0.0, 1.0),  # (B,H,W)
                "depth": out_depth  # (B,H,W)
            }
            if per_channel_sensitivity:
                if reduce_is_mean:
                    counts = gauss_batch_sens_count_ch
                    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
                    sens_channel = gauss_batch_sens_sum_ch / safe_counts
                    out["sens_channel"] = sens_channel.clamp(0.0, 1.0)
                    out["sens_channel_count"] = counts
                else:
                    out["sens_channel"] = gauss_batch_sens_max_ch.clamp(0.0, 1.0)
            if per_pixel_sensitivity:
                if reduce_is_mean:
                    counts = gauss_batch_sens_count_pix
                    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
                    sens_pixel = gauss_batch_sens_sum_pix / safe_counts
                    out["sens_pixel"] = sens_pixel.clamp(0.0, 1.0)
                    out["sens_pixel_count"] = counts
                else:
                    out["sens_pixel"] = gauss_batch_sens_max_pix.clamp(0.0, 1.0)
            if per_tile_sensitivity:
                if reduce_is_mean:
                    counts = gauss_batch_sens_count_tile
                    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
                    sens_tile = gauss_batch_sens_sum_tile / safe_counts
                    out["sens_tile"] = sens_tile.clamp(0.0, 1.0)
                    out["sens_tile_count"] = counts
                else:
                    out["sens_tile"] = gauss_batch_sens_max_tile.clamp(0.0, 1.0)
            if per_image_sensitivity:
                denom_per_img = denom_curr_img_raw.clamp_min(torch_ctx.denom_eps)
                gauss_batch_ratio_curr_img = gauss_contrib_curr_img / denom_per_img  # (N,)
                if reduce_is_mean:
                    contrib_mask = (gauss_contrib_curr_img > 0).to(torch_ctx.dtype)
                    gauss_batch_sens_sum_img += gauss_batch_ratio_curr_img
                    gauss_batch_sens_count_img += contrib_mask
                    counts = gauss_batch_sens_count_img
                    safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
                    sens_image = gauss_batch_sens_sum_img / safe_counts
                    out["sens_image"] = sens_image.clamp(0.0, 1.0)
                    out["sens_image_count"] = counts
                else:
                    gauss_batch_sens_max_img = torch.maximum(
                        gauss_batch_sens_max_img, gauss_batch_ratio_curr_img
                    )
                    out["sens_image"] = gauss_batch_sens_max_img.clamp(0.0, 1.0)
            if sens_need_per_batch:
                gauss_contrib_per_batch += gauss_contrib_curr_img
                denom_per_batch_raw += denom_curr_img_raw
            if per_batch_sensitivity:
                denom_per_batch = denom_per_batch_raw.clamp_min(torch_ctx.denom_eps)
                gauss_batch_sens_max_batch = gauss_contrib_per_batch / denom_per_batch
                out["sens_batch"] = gauss_batch_sens_max_batch.clamp(0.0, 1.0)
            if per_scene_sensitivity:
                out["nom_batch"] = gauss_contrib_per_batch
                out["denom_batch_raw"] = denom_per_batch_raw
            return out


    @torch.inference_mode()
    def render_batch_no_sens(self,
                     model_ply: GaussianPLY) -> dict:
        return self.render_batch(model_ply,
                                 per_channel_sensitivity=False,
                                 per_pixel_sensitivity=False,
                                 per_tile_sensitivity=False,
                                 per_image_sensitivity=False,
                                 per_batch_sensitivity=False,
                                 per_scene_sensitivity=False)


def iter_render_batches(
    model_ply: GaussianPLY,
    cameras: Sequence[Camera],
    *,
    batch_size: Optional[int],
    device: Optional[torch.device],
    block_xy: int = 16,
    white_background: bool = False,
    bg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    render_kwargs: Optional[Dict[str, Any]] = None,
) -> Iterator[Tuple[int, int, List[Camera], Dict[str, torch.Tensor], float]]:
    """
    Yield batched renderer outputs for `cameras` using `BatchedRenderer`.

    Returns tuples of (batch_id, camera_start_index, batch_cameras, outputs, elapsed_seconds).
    """
    total = len(cameras)
    if total == 0:
        return

    chunk = batch_size or total
    render_kwargs = render_kwargs or {}
    cam_idx = 0
    batch_id = 0
    dev = device or torch_ctx.device

    while cam_idx < total:
        batch_cams = list(cameras[cam_idx: cam_idx + chunk])
        renderer = BatchedRenderer(
            batch_cams,
            bg=bg,
            device=dev,
            white_background=white_background,
            block_xy=block_xy,
        )
        t0 = time.perf_counter()
        outputs = renderer.render_batch(model_ply=model_ply, **render_kwargs)
        elapsed = time.perf_counter() - t0
        yield batch_id, cam_idx, batch_cams, outputs, elapsed
        cam_idx += len(batch_cams)
        batch_id += 1


def save_render_batch_images(
    outputs: Dict[str, torch.Tensor],
    out_dir: str | Path,
    *,
    batch_id: int,
    camera_start_index: int,
    stem: str,
    save_rgb: bool = True,
    save_alpha: bool = False,
    save_depth: bool = False,
) -> None:
    """
    Persist RGB / alpha / depth tensors from a renderer batch.
    """
    if not (save_rgb or save_alpha or save_depth):
        return

    from torchvision.utils import save_image  # local import avoids hard dependency elsewhere

    base_dir = Path(out_dir)
    ensure_dir_for(base_dir, treat_as_file=False)

    rgb = outputs.get("rgb")
    alpha = outputs.get("alpha")
    depth = outputs.get("depth")

    if save_rgb and rgb is None:
        raise KeyError("Expected 'rgb' in renderer outputs when save_rgb=True")
    if save_alpha and alpha is None:
        raise KeyError("Expected 'alpha' in renderer outputs when save_alpha=True")
    if save_depth and depth is None:
        raise KeyError("Expected 'depth' in renderer outputs when save_depth=True")

    batch_size = rgb.shape[0] if rgb is not None else (
        alpha.shape[0] if alpha is not None else depth.shape[0]
    )

    rgb_cpu = rgb.detach().clamp(0, 1).cpu() if rgb is not None else None
    alpha_cpu = alpha.detach().clamp(0, 1).cpu() if alpha is not None else None
    depth_cpu = depth.detach().cpu() if depth is not None else None

    for local_idx in range(batch_size):
        global_idx = camera_start_index + local_idx
        if save_rgb and rgb_cpu is not None:
            rgb_t = rgb_cpu[local_idx].permute(2, 0, 1).contiguous()
            save_image(
                rgb_t,
                base_dir / f"{stem}_rgb_b{batch_id}_c{global_idx}.png",
            )
        if save_alpha and alpha_cpu is not None:
            alpha_t = alpha_cpu[local_idx].unsqueeze(0).contiguous()
            save_image(
                alpha_t,
                base_dir / f"{stem}_alpha_b{batch_id}_c{global_idx}.png",
            )
        if save_depth and depth_cpu is not None:
            depth_im = depth_cpu[local_idx]
            dmin = depth_im.min()
            dmax = depth_im.max()
            depth_norm = (depth_im - dmin) / (dmax - dmin + 1e-8)
            depth_t = depth_norm.unsqueeze(0).contiguous()
            save_image(
                depth_t,
                base_dir / f"{stem}_depth_b{batch_id}_c{global_idx}.png",
            )
