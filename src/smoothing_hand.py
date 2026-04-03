from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm

__all__ = ['remove_humps_and_smooth', 'compute_vertex_saliency']

# ─────────── user parameters ────────────
IN_MESH        = Path(r"C:\Users\alexk\Desktop\arm_only.obj")
OUT_COLOURED   = Path(r"C:\Users\alexk\Desktop\arm_saliency_colored.obj")
OUT_CLEANED    = Path(r"C:\Users\alexk\Desktop\arm_cleaned.obj")
DATA_NPZ       = Path(r"C:\Users\alexk\Desktop\arm_saliency_data.npz")

KEEP_QUANTILE  = 0.25   # keep lowest-25% saliency (dark-blue)
EPS            = 1e-12
# ────────────────────────────────────────

def compute_vertex_saliency(mesh: o3d.geometry.TriangleMesh,
                            n_levels: int = 6) -> np.ndarray:
    """Lee-style multi-scale curvature/normal-variation saliency."""
    verts   = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    tree    = cKDTree(verts)

    n        = len(verts)
    saliency = np.zeros(n)
    diag     = np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound())
    base_rad = 0.003 * diag

    for L in range(n_levels):
        r1 = base_rad * (L + 2)
        r2 = 2 * r1
        s1 = np.zeros(n)
        s2 = np.zeros(n)

        for i in range(n):
            idx1 = tree.query_ball_point(verts[i], r1)
            idx2 = tree.query_ball_point(verts[i], r2)

            if idx1:
                na1 = normals[idx1].mean(axis=0)
                na1 /= (np.linalg.norm(na1) + EPS)
                s1[i] = 1.0 - normals[i].dot(na1)
            if idx2:
                na2 = normals[idx2].mean(axis=0)
                na2 /= (np.linalg.norm(na2) + EPS)
                s2[i] = 1.0 - normals[i].dot(na2)

        diff = np.abs(s1 - s2)
        if diff.max() > 0:
            diff /= diff.max()
        saliency += diff

    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency

def visualize_saliency(saliency: np.ndarray) -> None:
    plt.figure(figsize=(10, 3))
    plt.hist(saliency, bins=100, color='orange', edgecolor='black')
    plt.title("Vertex Saliency Histogram")
    plt.xlabel("Saliency Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def remove_humps_and_smooth(
    mesh: o3d.geometry.TriangleMesh,
    keep_quantile: float = KEEP_QUANTILE,
    poisson_depth: int = 9,
    poisson_scale: float = 1.1,
    taubin_iters: int = 5,
    simple_iters: int = 2
) -> o3d.geometry.TriangleMesh:

    # 1) Saliency
    sal = compute_vertex_saliency(mesh)
    thresh = np.quantile(sal, keep_quantile)
    dark_mask = sal <= thresh

    # 2) Remove high-saliency faces
    triangles = np.asarray(mesh.triangles)
    face_mask = np.all(dark_mask[triangles], axis=1)
    clean = o3d.geometry.TriangleMesh(mesh)
    clean.remove_triangles_by_mask(~face_mask)
    clean.remove_unreferenced_vertices()
    clean.compute_vertex_normals()

    # 3) Poisson reconstruction
    pcd = o3d.geometry.PointCloud()
    pcd.points = clean.vertices
    pcd.normals = clean.vertex_normals
    poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth,
        scale=poisson_scale,
        linear_fit=False
    )

    # 4) Crop to original region
    bbox = clean.get_axis_aligned_bounding_box()
    final = poisson_mesh.crop(bbox)

    # 5) Smoothing
    final = final.filter_smooth_taubin(number_of_iterations=taubin_iters)
    final.compute_vertex_normals()
    final = final.filter_smooth_simple(number_of_iterations=simple_iters)
    final.compute_vertex_normals()

    return final

def main() -> None:
    # step 1: load
    print(f"► reading mesh: {IN_MESH}")
    mesh = o3d.io.read_triangle_mesh(str(IN_MESH))
    if mesh.is_empty():
        sys.exit("✗ could not read input mesh")

    # step 2: saliency + histogram
    print("▶ computing vertex saliency…")
    sal = compute_vertex_saliency(mesh)
    visualize_saliency(sal)

    # step 3: colour by saliency and write preview
    colours  = cm.plasma(sal)[:, :3]
    mesh_col = o3d.geometry.TriangleMesh(mesh)
    mesh_col.vertex_colors = o3d.utility.Vector3dVector(colours)
    print(f"▶ writing coloured saliency mesh → {OUT_COLOURED}")
    o3d.io.write_triangle_mesh(str(OUT_COLOURED), mesh_col)

    # step 4: remove humps and smooth using helper function
    print("▶ removing humps and smoothing mesh…")
    cleaned = remove_humps_and_smooth(
        mesh,
        keep_quantile=KEEP_QUANTILE,
        poisson_depth=9,
        poisson_scale=1.1,
        taubin_iters=5,
        simple_iters=2
    )

    # write final cleaned mesh
    print(f"▶ writing cleaned mesh → {OUT_CLEANED}")
    o3d.io.write_triangle_mesh(str(OUT_CLEANED), cleaned)

    # step 5: save raw saliency data
    dark_mask = sal <= np.quantile(sal, KEEP_QUANTILE)
    np.savez(str(DATA_NPZ),
             sal       = sal.astype(np.float32),
             rgb       = (colours * 255).astype(np.uint8),
             dark_mask = dark_mask)
    print(f"▶ saved saliency data → {DATA_NPZ}")

    print(f"✓ all done. cleaned mesh: {len(cleaned.vertices):,} V / {len(cleaned.triangles):,} F")


if __name__ == "__main__":
    main()
