import argparse
import os

import numpy as np
import open3d as o3d
import scipy.sparse
import scipy.sparse.linalg
from sklearn.cluster import KMeans
import copy
from collections import defaultdict


from scipy.spatial import cKDTree

from body_parts_allocation import (
    group_segments_by_body_parts,
    get_body_part_vertices,
    centralize_mesh_upright
)

__all__ = ['segment_human_mesh', 'color_mesh_by_labels']

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
###############################################################################
# 1) Basic Adjacency (used for concavity checks)
###############################################################################

def build_vertex_adjacency(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n = len(verts)
    adjacency = [set() for _ in range(n)]
    for tri in tris:
        i0, i1, i2 = tri
        adjacency[i0].add(i1);
        adjacency[i0].add(i2)
        adjacency[i1].add(i0);
        adjacency[i1].add(i2)
        adjacency[i2].add(i0);
        adjacency[i2].add(i1)
    return adjacency


###############################################################################
# 2) Standard Cotangent Laplacian & Mean Curvature
###############################################################################

def angle_between_vectors(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoids domain error in arccos
    return np.arccos(cos_theta)

def compute_mixed_voronoi_area(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n = len(verts)
    A = np.zeros(n)

    for tri in tris:
        i0, i1, i2 = tri
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]

        # Edge vectors
        e0 = p1 - p0
        e1 = p2 - p1
        e2 = p0 - p2

        # Edge lengths squared
        l0 = np.dot(e0, e0)
        l1 = np.dot(e1, e1)
        l2 = np.dot(e2, e2)

        # Compute angles safely
        angle0 = angle_between_vectors(-e2, e0)
        angle1 = angle_between_vectors(-e0, e1)
        angle2 = angle_between_vectors(-e1, e2)

        # Triangle area
        area = 0.5 * np.linalg.norm(np.cross(e0, -e2))

        # Check for obtuse angles
        if angle0 > np.pi / 2:
            A[i0] += area / 2
            A[i1] += area / 4
            A[i2] += area / 4
        elif angle1 > np.pi / 2:
            A[i1] += area / 2
            A[i0] += area / 4
            A[i2] += area / 4
        elif angle2 > np.pi / 2:
            A[i2] += area / 2
            A[i0] += area / 4
            A[i1] += area / 4
        else:
            # Safely compute cotangents
            tan0 = np.tan(angle0)
            tan1 = np.tan(angle1)
            tan2 = np.tan(angle2)

            cot0 = 1.0 / (tan0 + 1e-12)
            cot1 = 1.0 / (tan1 + 1e-12)
            cot2 = 1.0 / (tan2 + 1e-12)

            A[i0] += (l1 * cot2 + l2 * cot1) / 8
            A[i1] += (l2 * cot0 + l0 * cot2) / 8
            A[i2] += (l0 * cot1 + l1 * cot0) / 8

    return A

def build_cotangent_laplacian(mesh):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n = len(verts)

    # (1) Accurate Voronoi areas (Mixed Voronoi Formula)
    A = compute_mixed_voronoi_area(mesh)  # << must use MIXED voronoi area!

    # (2) Build edge -> incident faces map
    edge_to_faces = defaultdict(list)
    for fi, (i, j, k) in enumerate(tris):
        for a, b in ((i, j), (j, k), (k, i)):
            key = tuple(sorted((a, b)))
            edge_to_faces[key].append(fi)

    # (3) Cotangent weight function
    def compute_cotangent(vid, face_idx):
        tri = tris[face_idx]
        if vid not in tri:
            return 0.0
        i0, i1, i2 = tri
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        if vid == i0:
            vA, vB = p1, p2
        elif vid == i1:
            vA, vB = p0, p2
        else:
            vA, vB = p0, p1
        va = vA - verts[vid]
        vb = vB - verts[vid]
        cross = np.linalg.norm(np.cross(va, vb))
        dot = np.dot(va, vb)
        return (dot / cross) if cross > 1e-12 else 0.0

    # (4) Build Laplacian entries
    adjacency = build_vertex_adjacency(mesh)
    row, col, val = [], [], []
    edges = [(i, j) for i in range(n) for j in adjacency[i] if j > i]

    for i, j in edges:
        faces = edge_to_faces.get((min(i, j), max(i, j)), [])
        cot_sum = sum(compute_cotangent(i, f) + compute_cotangent(j, f) for f in faces) * 0.5

        if A[i] > 1e-12:
            w_ij = cot_sum / A[i]
        else:
            w_ij = 0.0

        row += [i, j]
        col += [j, i]
        val += [w_ij, w_ij]

    from scipy.sparse import csr_matrix, diags
    W = csr_matrix((val, (row, col)), shape=(n, n))
    d = np.asarray(W.sum(axis=1)).flatten()
    D = diags(d, 0)
    L = D - W
    return L


def compute_mean_curvature(mesh):
    verts = np.asarray(mesh.vertices)
    n = len(verts)
    if n < 1:
        return np.zeros(n)
    L = build_cotangent_laplacian(mesh)
    X = verts  # shape (n, 3)
    LX = L.dot(X)
    mean_curv = 0.5 * np.linalg.norm(LX, axis=1)
    return mean_curv


###############################################################################
# 3) cKDTree Radius Search for Weighted Curvature
###############################################################################

def radius_weighted_curvature_kdtree(tree, all_verts, root_idx, mean_curv, scale_sq):
    import math
    root_pt = all_verts[root_idx]
    radius = math.sqrt(scale_sq)
    idxs = tree.query_ball_point(root_pt, radius)
    if not idxs:
        return mean_curv[root_idx]
    pts = all_verts[idxs]
    d2 = np.sum((pts - root_pt) ** 2, axis=1)
    w = np.exp(-d2 / (2 * scale_sq))
    weight_total = np.sum(w)
    if weight_total < 1e-12:
        return mean_curv[root_idx]
    weighted_sum = np.sum(w * mean_curv[idxs])
    return weighted_sum / weight_total


###############################################################################
# 4) Multi-Scale Saliency using cKDTree
###############################################################################

def compute_saliency_lee_kdtree(mesh, n_levels=8):
    """
    We increased n_levels from 5 to 8 for more detail.
    """
    verts = np.asarray(mesh.vertices)
    n = len(verts)
    if n < 1:
        return np.zeros(n)
    mean_curv = compute_mean_curvature(mesh)
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    diag_len = np.linalg.norm(bb_max - bb_min)
    extent = 0.003 * diag_len

    tree = cKDTree(verts)

    adjacency = build_vertex_adjacency(mesh)
    saliency = np.zeros(n)

    for level_i in range(n_levels):
        scale_factor = (level_i + 2)
        scale_sq = (scale_factor * extent) ** 2
        levelSal = np.zeros(n)
        sumSal = 0.0
        for v_idx in range(n):
            wC1 = radius_weighted_curvature_kdtree(tree, verts, v_idx, mean_curv, scale_sq)
            wC2 = radius_weighted_curvature_kdtree(tree, verts, v_idx, mean_curv, 4 * scale_sq)
            diff = abs(wC1 - wC2)
            levelSal[v_idx] = diff
            sumSal += diff
        if sumSal > 1e-12:
            levelSal /= sumSal
        maxLev = levelSal.max()

        # local maxima
        localmax_vals = []
        for v_idx in range(n):
            val = levelSal[v_idx]
            # local max if val >= neighbors
            is_peak = all(levelSal[nb] <= val for nb in adjacency[v_idx])
            # skip if it's the absolute max
            if is_peak and abs(val - maxLev) > 1e-12:
                localmax_vals.append(val)

        meanLocalMax = np.mean(localmax_vals) if localmax_vals else 0.0
        suppressionFactor = (maxLev - meanLocalMax) ** 2
        saliency += levelSal * suppressionFactor

    mn, mx = saliency.min(), saliency.max()
    if mx > mn:
        saliency = (saliency - mn) / (mx - mn)
    else:
        saliency[:] = 0.5
    return saliency


###############################################################################
# 5) Concavity Detection & Saliency Laplacian Construction
###############################################################################

def detect_concavity(mesh, adjacency, eps=0.005):
    """
    We lower eps from 0.01 to 0.005.


    This means smaller angles can now count as 'concave,'
    which might help separate the shoulders/arms from torso.
    """
    mesh.compute_vertex_normals()
    vnorms = np.asarray(mesh.vertex_normals)
    verts = np.asarray(mesh.vertices)
    n = len(verts)
    concave_mask = np.zeros(n, dtype=bool)

    def find_k_ring_set(start_idx, k=2):
        visited = set()
        frontier = {start_idx}
        for _ in range(k):
            next_frontier = set()
            for v in frontier:
                visited.add(v)
                for w in adjacency[v]:
                    if w not in visited:
                        next_frontier.add(w)
            frontier = next_frontier
        return visited

    ring2 = []
    for i in range(n):
        r2 = find_k_ring_set(i, k=2)
        nf = np.mean(vnorms[list(r2)], axis=0)
        nf /= (np.linalg.norm(nf) + 1e-12)
        ring2.append(nf)

    for i in range(n):
        ni = vnorms[i]
        edge_concave = False
        for j in adjacency[i]:
            vij = verts[j] - verts[i]
            dist = np.linalg.norm(vij)
            if dist < 1e-12:
                continue
            vij_unit = vij / dist
            diff_normal = vnorms[j] - ni
            dd = vij_unit.dot(diff_normal)
            if dd > eps:
                edge_concave = True
                break
        nf = ring2[i]
        dd2 = nf.dot(ni)
        face_concave = (1.0 - dd2 > eps)
        if edge_concave and face_concave:
            concave_mask[i] = True
    return concave_mask


def build_saliency_laplacian(mesh, saliency, adjacency, concavity_mask):
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    n = len(verts)
    A = np.zeros(n)
    for tri in tris:
        i0, i1, i2 = tri
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        A[i0] += area / 3.0
        A[i1] += area / 3.0
        A[i2] += area / 3.0
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(tris):
        s = sorted(tri)
        edge_to_faces[(s[0], s[1])].append(fi)
        edge_to_faces[(s[0], s[2])].append(fi)
        edge_to_faces[(s[1], s[2])].append(fi)

    def compute_cot_angle_in_face(vid, face_idx):
        tri_ = tris[face_idx]
        if vid not in tri_:
            return 0.0
        i0, i1, i2 = tri_
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        if vid == i0:
            vA, vB = p1, p2
        elif vid == i1:
            vA, vB = p0, p2
        else:
            vA, vB = p0, p1
        va = vA - verts[vid]
        vb = vB - verts[vid]
        cross_ = np.linalg.norm(np.cross(va, vb))
        dot_ = va.dot(vb)
        if cross_ < 1e-12:
            return 0.0
        return dot_ / cross_

    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if j > i:
                edges.append((i, j))

    dij_vals = []
    for (i, j) in edges:
        sal_part = abs(saliency[i] - saliency[j])
        eta = 0.8 if (concavity_mask[i] and concavity_mask[j]) else 0.2
        i_, j_ = min(i, j), max(i, j)
        faces_ = edge_to_faces[(i_, j_)]
        cot_sum = 0.0
        for fID in faces_:
            c_i = compute_cot_angle_in_face(i, fID)
            c_j = compute_cot_angle_in_face(j, fID)
            cot_sum += (c_i + c_j)
        cot_sum *= 0.5
        A_i = A[i]
        if A_i < 1e-12:
            geo_part = 0.0
        else:
            dist_ij = np.linalg.norm(verts[j] - verts[i])
            geo_part = (cot_sum / A_i) * dist_ij
        d_ij = sal_part + eta * geo_part
        dij_vals.append(d_ij)

    if not dij_vals:
        raise ValueError("No edges found, or mesh is empty")
    sigma = max(dij_vals)

    rowL, colL, valL = [], [], []
    idx = 0
    for (i, j) in edges:
        d_ij = dij_vals[idx]
        w_ij = np.exp(-d_ij / (2 * sigma * sigma))
        rowL.extend([i, j])
        colL.extend([j, i])
        valL.extend([w_ij, w_ij])
        idx += 1

    from scipy.sparse import csr_matrix, diags
    W = csr_matrix((valL, (rowL, colL)), shape=(n, n))
    dvals = np.asarray(W.sum(axis=1)).flatten()
    D = diags(dvals, 0)
    L = D - W
    return L


def laplacian_eig_decompose(L, kmax=30):
    n = L.shape[0]
    k = min(kmax + 1, n - 2)
    evals, evecs = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def compute_gps(evals, evecs, skip=1, dim=None):
    n, nEV = evecs.shape
    if dim is None:
        dim = nEV - skip
    out = []
    for d in range(skip, skip + dim):
        lam = evals[d]
        if lam < 1e-12:
            lam = 1e-12
        out.append(evecs[:, d] / np.sqrt(lam))
    return np.array(out).T


def compute_hks(evals, evecs, t=1.5, skip=1, dim=None):
    n, nEV = evecs.shape
    if dim is None:
        dim = nEV - skip
    out = []
    for d in range(skip, skip + dim):
        lam = evals[d]
        coeff = np.exp(-lam * t)
        comp = coeff * (evecs[:, d] ** 2)
        out.append(comp)
    return np.array(out).T


def pick_num_segments(evals, max_k=15, start=1):
    """
    Determines the number of segments based on the largest eigenvalue gap,
    ensuring at least 6 segments.
    """
    diffs = []
    for i in range(start, len(evals) - 1):
        gap = evals[i + 1] - evals[i]
        diffs.append((gap, i))
    if not diffs:
        return 6  # fallback to minimum of 6 segments
    diffs.sort(key=lambda x: x[0], reverse=True)
    best_gap, best_i = diffs[0]
    # Ensure at least 6 segments
    k = max(best_i, 6)
    k = min(k, max_k)  # also clamp to max_k
    return k


###############################################################################
# 6) FULL SEGMENTATION PIPELINE
###############################################################################

def segment_human_mesh(mesh, clas=None):
    # (1) Compute multi-scale saliency using cKDTree with 8 levels
    saliency = compute_saliency_lee_kdtree(mesh, n_levels=8)

    # (2) Build adjacency for concavity check
    adjacency = build_vertex_adjacency(mesh)

    # (3) Detect concavity with eps=0.005
    concavity_mask = detect_concavity(mesh, adjacency, eps=0.005)

    # (4) Build the saliency Laplacian
    L = build_saliency_laplacian(mesh, saliency, adjacency, concavity_mask)

    # (5) Eigen-decompose L
    evals, evecs = laplacian_eig_decompose(L, kmax=30)

    # (6) Pick number of segments if not provided
    if clas is None:
        clas = pick_num_segments(evals, max_k=15, start=1)

    # (7) Compute GPS and HKS descriptors
    gps = compute_gps(evals, evecs, skip=1, dim=clas)
    hks = compute_hks(evals, evecs, t=1.5, skip=1, dim=clas)
    combo = 0.7 * gps + 0.3 * hks

    # (8) K-means with n_clusters = clas
    kmeans = KMeans(n_clusters=clas, n_init=10)
    labels = kmeans.fit_predict(combo)
    return labels


def color_mesh_by_labels(mesh, labels):
    # 13 named colors in a fixed order
    color_names = [
        'dark_blue', 'red', 'pink', 'yellow', 'green', 'orange', 'purple',
        'brown', 'light_blue', 'white', 'black', 'cyan', 'magenta'
    ]
    rgb_list = [
        [0.0, 0.0, 0.5], [1.0, 0.0, 0.0], [1.0, 0.75, 0.8], [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.6, 0.3, 0.0],
        [0.68, 0.85, 0.9], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0]
    ]
    K = max(labels) + 1
    assert K <= len(rgb_list), "Not enough predefined colors!"

    # assign to vertices
    vcolors = np.array([rgb_list[lbl] for lbl in labels])
    meshc = copy.deepcopy(mesh)
    meshc.vertex_colors = o3d.utility.Vector3dVector(vcolors)

    # return also the two lists for downstream
    return meshc, color_names, rgb_list



###############################################################################
# 7) MAIN CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Faster multi-scale saliency segmentation (cKDTree) of a human OBJ.")
    parser.add_argument("--input", default=r"C:\Users\alexk\Desktop\testObj\5.obj", help="Path to input .obj")
    parser.add_argument("--output_segmented", default="segmented.obj", help="Output path for colored .obj")
    parser.add_argument("--output_with_markers",
                        default="segmented_with_markers.obj",
                        help="Output path for mesh + marker spheres")
    parser.add_argument("--output_smoothed", default="body_smoothed.obj",
                        help="Output path for smoothed body OBJ")
    parser.add_argument("--output_head", default="head_only.obj",
                        help="Output path for head-only OBJ")
    parser.add_argument("--output_arm", default="arm_only.obj",
                        help="Output path for left hand OBJ")
    args = parser.parse_args()

    # 1) Load
    mesh = o3d.io.read_triangle_mesh(args.input)
    if mesh.is_empty():
        print("Failed to load mesh:", args.input)
        return

    mesh = centralize_mesh_upright(mesh)

    # 2) Segment
    labels = segment_human_mesh(mesh, clas=13)
    print("Done segmentation. #clusters =", np.max(labels) + 1)

    # 3) Color
    # grab the same named_colors dict you used inside color_mesh_by_labels
    colored_mesh, color_names, rgb_list = color_mesh_by_labels(mesh, labels)
    o3d.io.write_triangle_mesh(args.output_segmented, colored_mesh)

    # 4) Group segments into body parts
    segments = [np.where(labels == k)[0] for k in range(labels.max() + 1)]
    body_parts = group_segments_by_body_parts(segments, mesh)

    # —————————————————————————————————————————————
    # Print each part’s cluster IDs *and* its assigned colors
    print("\nBody Parts and their colors:")
    for part, seg_ids in body_parts.items():
        if not seg_ids:
            print(f"  {part:12s}: (no segments)")
            continue
        names = [color_names[sid] for sid in seg_ids]
        cols = [rgb_list[sid] for sid in seg_ids]
        print(f"  {part:12s}: clusters {seg_ids}")
        print(f"    → color names = {names}")
        print()

        # 5) Print & build marker spheres
    palette = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0], [0, 0.5, 0.5]
    ]
    spheres = []
    aabb = colored_mesh.get_axis_aligned_bounding_box()
    radius = np.linalg.norm(aabb.get_max_bound() - aabb.get_min_bound()) * 0.01

    # 4) Grab per-part data (incl. vertices, begin & end)
    info = get_body_part_vertices(body_parts, segments, mesh)

    for idx, part in enumerate(body_parts):
        data = info[part]
        pts = data['vertices']
        bpt = data['begin']
        ept = data['end']

        print(f"{part}: #verts={pts.shape[0]}",
              f"begin={None if bpt is None else bpt.tolist()}",
              f"end={None if ept is None else ept.tolist()}")

        if bpt is None or ept is None:
            continue

        color = palette[idx % len(palette)]
        for P in (bpt, ept):
            s = o3d.geometry.TriangleMesh.create_sphere(
                radius=radius, resolution=10)
            s.paint_uniform_color(color)
            s.translate(P)
            spheres.append(s)

    # 6) Merge markers and write
    combined = copy.deepcopy(colored_mesh)
    for sph in spheres:
        combined += sph

    o3d.io.write_triangle_mesh(args.output_with_markers, combined)
    print("Wrote mesh with markers to", args.output_with_markers)



if __name__ == "__main__":
    main()

