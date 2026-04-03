import numpy as np
from collections import defaultdict
from itertools import combinations
import open3d as o3d

__all__ = ['group_segments_by_body_parts', 'get_body_part_vertices', 'get_body_part_boundary_vertices', 'align_vertices_to_vertical', 'centralize_mesh_upright']

def align_vertices_to_vertical(verts: np.ndarray):
    """
    Aligns a set of vertices to the vertical axis (Y-axis) using PCA.
    Returns the aligned vertices, rotation matrix, centroid, and minimum Y value.
    """
    # Center vertices at centroid
    centroid = verts.mean(axis=0)
    Vc = verts - centroid

    # PCA to find principal axis
    cov = np.cov(Vc.T)
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, np.argmax(evals)]
    # Ensure principal axis points upward
    if principal[1] < 0:
        principal = -principal

    # Rodrigues formula to rotate principal axis to +Y
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(principal, target)
    s = np.linalg.norm(v)
    c = np.dot(principal, target)
    if s > 1e-6:
        K = np.array([[    0, -v[2],  v[1]],
                      [ v[2],     0, -v[0]],
                      [-v[1],  v[0],     0]])
        R = np.eye(3) + K + (K @ K) * ((1 - c) / (s * s))
    else:
        R = np.eye(3)

    # Apply rotation
    rotated = Vc @ R.T

    # Shift so minimum Y is 0
    y_min = rotated[:,1].min()
    rotated[:,1] -= y_min

    return rotated, R, centroid, y_min

def centralize_mesh_upright(mesh):
    """
    Centralizes and uprights a mesh using PCA alignment.
    Returns the transformed mesh.
    """
    verts = np.asarray(mesh.vertices, float)

    # Align vertices to vertical
    aligned, R, centroid, _ = align_vertices_to_vertical(verts)

    # Check if mesh is upside-down
    y = aligned[:,1]
    height = y.max() - y.min()
    slice_h = height * 0.05
    bottom_count = np.sum(y < y.min() + slice_h)
    top_count = np.sum(y > y.max() - slice_h)

    if top_count > bottom_count:
        aligned[:,1] = -aligned[:,1]

    # Shift so minimum Y is 0
    aligned[:,1] -= aligned[:,1].min()

    # Create new mesh
    m2 = o3d.geometry.TriangleMesh()
    m2.vertices = o3d.utility.Vector3dVector(aligned)
    m2.triangles = mesh.triangles
    if mesh.has_vertex_normals():
        m2.vertex_normals = mesh.vertex_normals
    if mesh.has_vertex_colors():
        m2.vertex_colors  = mesh.vertex_colors
    return m2

def group_segments_by_body_parts(segments, mesh):
    """
    Groups mesh segments into body parts (head, torso, legs, arms).
    Returns a dictionary mapping body parts to segment indices.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.size == 0:
        return {}

    # Calculate segment centroids
    seg_centroids = []
    for seg in segments:
        if len(seg) == 0:
            seg_centroids.append((np.zeros(3), 0.0))
        else:
            cen = verts[seg].mean(axis=0)
            seg_centroids.append((cen, cen[1]))

    # Build vertex adjacency graph
    vert_neighbors = defaultdict(set)
    tris = np.asarray(mesh.triangles, dtype=int)
    for tri in tris:
        for u, v in combinations(tri, 2):
            vert_neighbors[u].add(v)
            vert_neighbors[v].add(u)
    seg_vertex_sets = [set(seg) for seg in segments]

    # Check segment connectivity
    def is_connected(i, j):
        return any(
            neigh in seg_vertex_sets[i]
            for v in seg_vertex_sets[j]
            for neigh in vert_neighbors[v]
        )

    # Initialize body parts dictionary
    used = set()
    parts = {p: [] for p in ('head','torso','left_leg','right_leg','left_arm','right_arm')}

    # Identify head segment (highest Y centroid)
    head_idx = int(np.argmax([y for _, y in seg_centroids]))
    parts['head'].append(head_idx)
    used.add(head_idx)

    # Identify torso segments (below head and connected)
    rem = [i for i in range(len(seg_centroids)) if i not in used]
    torso_cands = [
        i for i in rem
        if seg_centroids[i][1] < seg_centroids[head_idx][1]
           and is_connected(head_idx, i)
    ]
    if torso_cands:
        torso1 = max(torso_cands, key=lambda i: seg_centroids[i][1])
    else:
        torso1 = max(rem, key=lambda i: seg_centroids[i][1])
    parts['torso'].append(torso1)
    used.add(torso1)

    # Identify pelvis segment (connected to bottom of torso)
    rem = [i for i in range(len(seg_centroids)) if i not in used]
    torso_vids = segments[torso1]
    ys = verts[torso_vids, 1]
    torso_min_y = float(ys.min())
    bottom_ring = [v for v in torso_vids if abs(verts[v,1] - torso_min_y) < 1e-6]
    pelvis_cands = []
    for i in rem:
        cen, y = seg_centroids[i]
        if y >= torso_min_y:
            continue
        if not any(neigh in seg_vertex_sets[i] for v in bottom_ring for neigh in vert_neighbors[v]):
            continue
        if abs(cen[0]) > 0.6 * (verts[torso_vids,0].max() - verts[torso_vids,0].min()):
            continue
        pelvis_cands.append(i)
    if pelvis_cands:
        pelvis = min(pelvis_cands, key=lambda i: abs(seg_centroids[i][0][0]))
        parts['torso'].append(pelvis)
        used.add(pelvis)

    # Identify legs (two lowest segments)
    rem = [i for i in range(len(seg_centroids)) if i not in used]
    lowest_two = sorted(rem, key=lambda i: seg_centroids[i][1])[:2]
    if len(lowest_two) == 2:
        f1, f2 = lowest_two
        x1, x2 = seg_centroids[f1][0][0], seg_centroids[f2][0][0]
        parts['left_leg'].append(f1 if x1 > x2 else f2)
        parts['right_leg'].append(f2 if x1 > x2 else f1)
        used.update(lowest_two)

    # Grow legs upward
    def climb(prev):
        rem2 = [i for i in range(len(seg_centroids)) if i not in used]
        y0 = seg_centroids[prev][1]
        cands = []
        for i in rem2:
            cen, y = seg_centroids[i]
            if y <= y0 or not is_connected(prev, i):
                continue
            cands.append((np.linalg.norm(cen - seg_centroids[prev][0]), i))
        return min(cands)[1] if cands else None

    for side in ['left_leg', 'right_leg']:
        if not parts[side]:
            continue
        prev = parts[side][-1]
        for _ in range(2):
            if prev is None:
                break
            nxt = climb(prev)
            if nxt is None:
                break
            parts[side].append(nxt)
            used.add(nxt)
            prev = nxt

    # Identify arms (connected to torso)
    rem = [i for i in range(len(seg_centroids)) if i not in used]
    torso_idxs = parts['torso']
    for arm, sign in [('left_arm', -1), ('right_arm', 1)]:
        cands = [
            i for i in rem
            if seg_centroids[i][0][0] * sign > 0
               and any(is_connected(i, t) for t in torso_idxs)
        ]
        if cands:
            top = max(cands, key=lambda i: seg_centroids[i][1])
            parts[arm].append(top)
            used.add(top)
            rem2 = [i for i in range(len(seg_centroids)) if i not in used]
            sec = [i for i in rem2 if is_connected(top, i)]
            if sec:
                parts[arm].append(max(sec, key=lambda i: seg_centroids[i][1]))
                used.add(max(sec, key=lambda i: seg_centroids[i][1]))

    # Handle remaining segments
    left_assigned = bool(parts['left_arm'])
    right_assigned = bool(parts['right_arm'])
    rem = [i for i in range(len(seg_centroids)) if i not in used]

    if left_assigned ^ right_assigned and len(rem) >= 2:
        by_height = sorted(rem, key=lambda i: seg_centroids[i][1], reverse=True)
        fallback = by_height[:2]
        side = 'left_arm' if not left_assigned else 'right_arm'
        for i in fallback:
            parts[side].append(i)
            used.add(i)

    return parts

def get_body_part_vertices(parts, segments, mesh):
    """
    Retrieves vertices for each body part and their boundary points.
    Returns a dictionary with vertices, begin, and end points for each part.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.size == 0:
        return {p: {'vertices': np.zeros((0,3)), 'begin': None, 'end': None}
                for p in parts}

    # Precompute segment Y-bounds
    seg_min = []
    seg_max = []
    for seg in segments:
        if len(seg) == 0:
            seg_min.append(np.inf)
            seg_max.append(-np.inf)
        else:
            ys = verts[seg, 1]
            seg_min.append(float(ys.min()))
            seg_max.append(float(ys.max()))

    out = {}
    for part, seg_ids in parts.items():
        if not seg_ids:
            out[part] = {'vertices': np.zeros((0,3)), 'begin': None, 'end': None}
            continue

        # Collect vertices for this part
        vids = np.unique(np.concatenate([segments[s] for s in seg_ids]))
        part_pts = verts[vids]

        # Find bottom-most and top-most points
        low_seg_idx = int(np.argmin([seg_min[s] for s in seg_ids]))
        high_seg_idx = int(np.argmax([seg_max[s] for s in seg_ids]))
        low_seg = seg_ids[low_seg_idx]
        high_seg = seg_ids[high_seg_idx]

        low_pts = verts[segments[low_seg]]
        high_pts = verts[segments[high_seg]]
        b_pt = low_pts[np.argmin(low_pts[:,1])]
        e_pt = high_pts[np.argmax(high_pts[:,1])]

        out[part] = {
            'vertices': part_pts,
            'begin': b_pt,
            'end': e_pt
        }

    return out

def get_body_part_boundary_vertices(parts, segments, mesh):
    """
    Finds vertices on the boundary between different body parts.
    Returns a dictionary with boundary vertices for each part.
    """
    nverts = len(mesh.vertices)
    vert_to_part = -np.ones(nverts, dtype=int)
    for idx, name in enumerate(parts):
        for s in parts[name]:
            vert_to_part[segments[s]] = idx

    tris = np.asarray(mesh.triangles, dtype=int)
    adj = defaultdict(set)
    for tri in tris:
        for u, v in combinations(tri, 2):
            adj[u].add(v)
            adj[v].add(u)

    verts = np.asarray(mesh.vertices, dtype=float)
    boundary = {}
    for name, seg_ids in parts.items():
        vids = np.unique(np.concatenate([segments[s] for s in seg_ids]))
        bset = {v for v in vids if any(vert_to_part[w] != vert_to_part[v] for w in adj[v])}
        boundary[name] = verts[list(bset)] if bset else np.zeros((0,3))
    return boundary