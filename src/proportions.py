import numpy as np
import open3d as o3d
import copy
from scipy.spatial import ConvexHull, cKDTree
from body_parts_allocation import group_segments_by_body_parts, centralize_mesh_upright, get_body_part_vertices
from segmentation import segment_human_mesh

def _pick_chest_segment(mesh, parts, segments):
    """Return the torso segment with the highest centroid in Y."""
    verts = np.asarray(mesh.vertices)
    torso = parts.get('torso', [])
    if not torso:
        raise RuntimeError("No torso segments found.")
    best = (-np.inf, None)
    for s in torso:
        vids = segments[s]
        cy = verts[vids,1].mean() if len(vids)>0 else -np.inf
        if cy > best[0]:
            best = (cy, s)
    return best[1]

def detect_shoulder_endpoints(mesh, parts, segments, tol_slice=0.05, n_slices=20):
    """
    Returns two points spanning maximal shoulder width by analyzing horizontal slices.
    """
    verts = np.asarray(mesh.vertices)
    seg = _pick_chest_segment(mesh, parts, segments)
    pts = verts[segments[seg]]
    if pts.shape[0] < 2:
        raise RuntimeError("Not enough points for shoulders")

    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    tol_y = (y_max - y_min) * tol_slice
    ys = np.linspace(y_min + tol_y, y_max - tol_y, n_slices)

    best = (-np.inf, None, None, None)
    for y in ys:
        slc = pts[np.abs(pts[:,1] - y) <= tol_y]
        if slc.shape[0] < 2:
            continue

        # Calculate convex hull diameter in X-Z plane
        Q = slc[:, [0,2]]
        try:
            hull = ConvexHull(Q)
            hp = Q[hull.vertices]
            i_h, j_h = np.triu_indices(len(hp), 1)
            diffs = hp[i_h] - hp[j_h]
            d2_all = np.sum(diffs**2, axis=1)
            k = np.argmax(d2_all)
            i0, j0 = hull.vertices[i_h[k]], hull.vertices[j_h[k]]
        except:
            # Fallback to brute-force if convex hull fails
            n = len(slc)
            diffs = slc[:, None, :] - slc[None, :, :]
            d2mat = np.sum(diffs**2, axis=2)
            iu, ju = np.triu_indices(n, 1)
            if iu.size == 0:
                continue
            d2_all = d2mat[iu, ju]
            k = np.argmax(d2_all)
            i0, j0 = iu[k], ju[k]

        d2best = np.sum((slc[i0] - slc[j0]) ** 2)
        if d2best > best[0]:
            best = (d2best, slc[i0].copy(), slc[j0].copy(), y)

    _, pL, pR, y_slice = best
    pL[1] = pR[1] = y_slice  # Ensure same Y level
    return pL, pR

def detect_chest_endpoints(mesh, parts, segments, tol_slice=0.05, n_slices=20):
    """
    Returns front and back chest points using PCA analysis.
    """
    verts = np.asarray(mesh.vertices)
    seg = _pick_chest_segment(mesh, parts, segments)
    pts = verts[segments[seg]]
    if pts.shape[0] < 2:
        raise RuntimeError("Not enough chest points")

    Q_all = pts[:, [0,2]]
    C = np.cov(Q_all.T)
    ev, evec = np.linalg.eigh(C)
    axes = evec[:, np.argsort(ev)[::-1]]
    pc0, pc1 = axes[:,0], axes[:,1]

    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    tol_y = (y_max - y_min) * tol_slice
    ys = np.linspace(y_min + tol_y, y_max - tol_y, n_slices)

    best = (-np.inf, None, None, None, None)
    for y in ys:
        slc = pts[np.abs(pts[:,1] - y) <= tol_y]
        if slc.shape[0] < 2:
            continue

        Q = slc[:, [0,2]]
        center = Q.mean(axis=0)
        Qc = Q - center
        u = Qc.dot(pc0)
        v = Qc.dot(pc1)
        depth = v.max() - v.min()
        if depth > best[0]:
            best = (depth, slc, center, u, v, y)

    depth, slc, center, u, v, yb = best
    iF = np.argmax(v)
    pF = slc[iF].copy()
    pF[1] = yb

    # Calculate back point projection
    uF = u[iF]
    v_min = v.min()
    back_xz = center + uF*pc0 + v_min*pc1

    # Snap to nearest actual vertex
    tree = cKDTree(slc[:, [0,2]])
    _, idx = tree.query(back_xz)
    pB = slc[idx].copy()
    pB[1] = yb

    return pF, pB

def detect_waist_endpoints(mesh, parts, segments, tol_slice=0.05, n_slices=20, lower_frac=0.1, upper_frac=0.1):
    """
    Returns two points spanning the narrowest waist width.
    """
    verts = np.asarray(mesh.vertices)
    torso_segs = parts.get('torso', [])
    if len(torso_segs) < 2:
        raise RuntimeError("Need at least two torso segments for pelvis")

    # Pick pelvis segment (lowest centroid Y)
    best = (np.inf, None)
    for s in torso_segs:
        vids = segments[s]
        cy = verts[vids,1].mean() if len(vids)>0 else np.inf
        if cy < best[0]:
            best = (cy, s)
    pelvis_seg = best[1]
    pts = verts[segments[pelvis_seg]]
    if pts.shape[0] < 2:
        raise RuntimeError("Not enough points in pelvis segment")

    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    height = y_max - y_min
    band_min = y_min + height * lower_frac
    band_max = y_max - height * upper_frac
    tol_y = height * tol_slice
    ys = np.linspace(band_min, band_max, n_slices)

    best = (np.inf, None, None, None)
    for y in ys:
        slc = pts[np.abs(pts[:,1] - y) <= tol_y]
        if slc.shape[0] < 2:
            continue

        Q = slc[:, [0,2]]
        try:
            hull = ConvexHull(Q)
            hp = Q[hull.vertices]
            i_h, j_h = np.triu_indices(len(hp), 1)
            diffs = hp[i_h] - hp[j_h]
            d2_all = np.sum(diffs**2, axis=1)
            k = np.argmax(d2_all)
            i0, j0 = hull.vertices[i_h[k]], hull.vertices[j_h[k]]
        except:
            n = len(slc)
            diffs = slc[:, None, :] - slc[None, :, :]
            d2mat = np.sum(diffs**2, axis=2)
            iu, ju = np.triu_indices(n, 1)
            if iu.size == 0:
                continue
            d2_all = d2mat[iu, ju]
            k = np.argmax(d2_all)
            i0, j0 = iu[k], ju[k]

        d2best = np.sum((slc[i0] - slc[j0]) ** 2)
        if d2best < best[0]:  # We want the narrowest point
            best = (d2best, slc[i0].copy(), slc[j0].copy(), y)

    d2, pW_left, pW_right, y_slice = best
    pW_left[1] = pW_right[1] = y_slice
    return pW_left, pW_right

def detect_hip_depth_endpoints(mesh, parts, segments):
    verts = np.asarray(mesh.vertices)
    torso_ids = parts.get('torso', [])
    if len(torso_ids) < 2:
        raise RuntimeError("Need at least two torso segments for pelvis")

    # pick pelvis = lowest‐centroid torso segment
    best = (np.inf, None)
    for s in torso_ids:
        vids = segments[s]
        cy = verts[vids,1].mean() if len(vids)>0 else np.inf
        if cy < best[0]:
            best = (cy, s)
    pelvis_seg = best[1]
    pts = verts[segments[pelvis_seg]]
    if pts.shape[0] < 2:
        raise RuntimeError("Pelvis segment too small")

    # PCA on X–Z
    Q = pts[:, [0,2]]
    centre = Q.mean(axis=0)
    Qc = Q - centre
    evals, evecs = np.linalg.eigh(np.cov(Qc.T))
    # depth axis = minor principal direction
    pc1 = evecs[:, np.argmin(evals)]
    pc0 = evecs[:, np.argmax(evals)]

    # project onto depth
    v = Qc.dot(pc1)
    iB = np.argmin(v)
    pB = pts[iB].copy()
    y_slice = pB[1]

    # find the matching front point
    u_back   = Qc[iB].dot(pc0)
    v_front  = v.max()
    front_xz = centre + u_back*pc0 + v_front*pc1
    target3d = np.array([front_xz[0], y_slice, front_xz[1]])
    tree     = cKDTree(pts)
    _, iF    = tree.query(target3d)
    pF       = pts[iF].copy()
    pF[1]    = y_slice

    return pB, pF, y_slice


def detect_hip_width_endpoints(mesh, parts, segments, tol_slice=0.05):
    """
    Returns the two points spanning maximal hip width,
    forced to lie on the exact same horizontal plane as the hip-depth markers.
    tol_slice is the fraction of the pelvis height used as vertical tolerance.
    """
    # 0) First, get (back, front, y_slice) from your existing depth routine
    pB, pF, y_slice = detect_hip_depth_endpoints(mesh, parts, segments)

    # 1) Pick the same pelvis segment
    verts = np.asarray(mesh.vertices)
    torso_ids = parts.get('torso', [])
    if len(torso_ids) < 2:
        raise RuntimeError("Need at least two torso segments for pelvis")
    best = (np.inf, None)
    for s in torso_ids:
        vids = segments[s]
        cy = verts[vids,1].mean() if len(vids)>0 else np.inf
        if cy < best[0]:
            best = (cy, s)
    pelvis_seg = best[1]
    pts = verts[segments[pelvis_seg]]
    if pts.shape[0] < 2:
        raise RuntimeError("Pelvis segment too small for hip width")

    # 2) Build a thin horizontal band around y_slice
    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    tol_y = (y_max - y_min) * tol_slice
    band = pts[np.abs(pts[:,1] - y_slice) <= tol_y]
    if band.shape[0] < 2:
        raise RuntimeError("Not enough points at hip depth slice")

    # 3) Compute convex‐hull diameter in X–Z within that band
    Q = band[:, [0,2]]
    try:
        hull = ConvexHull(Q)
        hp = Q[hull.vertices]
        i_h, j_h = np.triu_indices(len(hp), 1)
        diffs = hp[i_h] - hp[j_h]
        k = np.argmax(np.sum(diffs**2, axis=1))
        # find which two hull‐vertices gave that max distance
        i0, j0 = hull.vertices[i_h[k]], hull.vertices[j_h[k]]
    except:
        # brute‐force fallback
        n = len(band)
        diffs = band[:, None, :] - band[None, :, :]
        d2mat = np.sum(diffs**2, axis=2)
        iu, ju = np.triu_indices(n, 1)
        if iu.size == 0:
            raise RuntimeError("Cannot compute hip width")
        k = np.argmax(d2mat[iu, ju])
        i0, j0 = iu[k], ju[k]

    # 4) Grab those two points and snap their Y to y_slice
    pL = band[i0].copy()
    pR = band[j0].copy()
    pL[1] = pR[1] = y_slice

    return pL, pR

def compute_forward_axis(mesh, parts, segments,
                         tol_slice: float = 0.05,
                         n_slices: int = 20):
    """
    Runs your existing chest‐depth routine to get pF (front) & pB (back),
    then returns the normalized 2D forward vector in the X–Z plane.
    """
    # you already have this function, imported:
    pF, pB = detect_chest_endpoints(mesh, parts, segments,
                                    tol_slice=tol_slice,
                                    n_slices=n_slices)
    fwd2 = pF[[0,2]] - pB[[0,2]]
    return fwd2 / np.linalg.norm(fwd2)


def detect_left_leg_deep_endpoints(
    mesh,
    parts,
    segments,
    which: str,
    forward2d: np.ndarray,
    tol_slice: float = 0.05,
    n_slices: int = 20,
    upper_frac: float = 0.3,
):
    """
    Over the left-leg sub-segment (upper or middle), but excluding its
    top `upper_frac`, find the horizontal slice whose front-back depth
    (along forward2d) is maximal. Returns the front and back points on
    that slice, snapped to the same Y level.
    """
    import numpy as np

    verts   = np.asarray(mesh.vertices)
    leg_ids = parts.get('left_leg', [])
    if not leg_ids:
        raise RuntimeError("No segments for 'left_leg'")

    # 1) pick the sub-segment by descending Y centroid
    cents = []
    for sid in leg_ids:
        vids = segments[sid]
        if vids.size == 0:
            continue
        cents.append((sid, verts[vids,1].mean()))
    if not cents:
        raise RuntimeError("No valid segments for 'left_leg'")
    cents.sort(key=lambda x: x[1], reverse=True)
    if which == 'upper':
        pick_sid = cents[0][0]
    elif which == 'middle':
        if len(cents) < 2:
            raise RuntimeError("Need at least two segments for middle slice")
        pick_sid = cents[1][0]
    else:
        raise ValueError("which must be 'upper' or 'middle'")

    # 2) extract pts and drop top `upper_frac`
    pts_all = verts[segments[pick_sid]]
    if pts_all.shape[0] < 2:
        raise RuntimeError(f"Segment {pick_sid} too small")

    y_min, y_max = pts_all[:,1].min(), pts_all[:,1].max()
    thresh_y     = y_min + (1.0 - upper_frac) * (y_max - y_min)
    mask         = pts_all[:,1] <= thresh_y
    pts = pts_all[mask]
    if pts.shape[0] < 2:
        raise RuntimeError("Not enough points after excluding top region")

    # 3) sweep horizontal slices to find maximal depth
    y_min_k, y_max_k = pts[:,1].min(), pts[:,1].max()
    height_k         = y_max_k - y_min_k
    tol_y_k          = tol_slice * height_k
    ys               = np.linspace(y_min_k + tol_y_k, y_max_k - tol_y_k, n_slices)

    best = (-np.inf, None, None, None)
    for y in ys:
        slc = pts[np.abs(pts[:,1] - y) <= tol_y_k]
        if slc.shape[0] < 2:
            continue
        Q2d   = slc[:, [0,2]]
        center= Q2d.mean(axis=0)
        projs = (Q2d - center).dot(forward2d)
        span  = projs.max() - projs.min()
        if span > best[0]:
            best = (span, slc, projs, y)

    span, slc, projs, yb = best
    if slc is None:
        raise RuntimeError("Left-leg deep detection failed: no valid slice found")

    # 4) choose front and back in that slice
    iF = np.argmax(projs)
    iB = np.argmin(projs)
    p_front = slc[iF].copy()
    p_back  = slc[iB].copy()

    # 5) snap to slice level Y
    p_front[1] = p_back[1] = yb
    return p_front, p_back



def compute_limb_lengths(mesh, parts, segments):
    """
    Calculates limb lengths and their ratios relative to model height.
    """
    bounds = mesh.get_axis_aligned_bounding_box()
    height = bounds.get_max_bound()[1] - bounds.get_min_bound()[1]

    pbv = get_body_part_vertices(parts, segments, mesh)

    out = {}
    for limb in ('left_arm', 'right_arm', 'left_leg', 'right_leg'):
        info = pbv.get(limb)
        if info is None or info['begin'] is None or info['end'] is None:
            out[limb] = {'length': None, 'ratio': None}
        else:
            b = info['begin']
            e = info['end']
            L = np.linalg.norm(e - b)
            out[limb] = {
                'length': L,
                'ratio': L / height if height > 0 else None
            }
    return out


def main():
    import numpy as np
    import open3d as o3d

    # read & normalize
    mesh = o3d.io.read_triangle_mesh(r"C:\Users\alexk\Desktop\testObj\10.obj")
    if mesh.is_empty():
        raise RuntimeError("Failed to load mesh")
    mesh = centralize_mesh_upright(mesh)

    # segmentation & grouping
    labels = segment_human_mesh(mesh, clas=13)
    segments = [np.where(labels == k)[0] for k in range(labels.max() + 1)]
    parts = group_segments_by_body_parts(segments, mesh)

    # torso landmarks
    pSL, pSR = detect_shoulder_endpoints(mesh, parts, segments)
    pCF, pCB = detect_chest_endpoints    (mesh, parts, segments)
    pWL, pWR = detect_waist_endpoints    (mesh, parts, segments)

    # hip depth & width
    pHB, pHF, hip_y = detect_hip_depth_endpoints(mesh, parts, segments)
    pHL, pHR       = detect_hip_width_endpoints(mesh, parts, segments, hip_y)

    # compute global forward direction from chest
    forward2d = compute_forward_axis(mesh, parts, segments)

    # leg‐deep endpoints (upper & middle) using forward2d
    pLU, pLBu = detect_left_leg_deep_endpoints(
        mesh, parts, segments,
        which='upper',
        forward2d=forward2d,
        upper_frac=0.3
    )

    # Mid‐thigh, skipping top 30%
    pLM, pLBm = detect_left_leg_deep_endpoints(
        mesh, parts, segments,
        which='middle',
        forward2d=forward2d,
        upper_frac=0.3
    )


    # compute bounding height
    aabb = mesh.get_axis_aligned_bounding_box()
    height = aabb.get_max_bound()[1] - aabb.get_min_bound()[1]

    # collect all marker pairs
    marker_groups = {
        'shoulder_width'   : (pSL, pSR),
        'chest_depth'      : (pCF, pCB),
        'waist_width'      : (pWL, pWR),
        'hip_depth'        : (pHB, pHF),
        'hip_width'        : (pHL, pHR),
        'left_leg_upper_d' : (pLU, pLBu),
        'left_leg_middle_d': (pLM, pLBm)

    }

    # print distances & ratios
    print("=== Marker Distances ===")
    for name, (A, B) in marker_groups.items():
        d = np.linalg.norm(B - A)
        print(f"{name:18s}: dist = {d:.3f}   ratio = {d/height:.3f}")

    # limb lengths
    limb_lengths = compute_limb_lengths(mesh, parts, segments)
    print("\n=== Limb Lengths ===")
    for limb, data in limb_lengths.items():
        L = data['length']; r = data['ratio']
        if L is None:
            print(f"{limb:12s}: length = None")
        else:
            print(f"{limb:12s}: length = {L:.3f}   ratio = {r:.3f}")

    # annotate mesh with spheres at each marker
    radius = height * 0.01
    annotated = mesh
    for A, B in marker_groups.values():
        for P in (A, B):
            sph = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=10)
            sph.translate(P)
            annotated += sph

    # write annotated OBJ
    out = r"C:\Users\alexk\Desktop\testObj\10_annotated.obj"
    o3d.io.write_triangle_mesh(out, annotated)
    print("\nWrote annotated mesh to", out)


if __name__ == "__main__":
    main()