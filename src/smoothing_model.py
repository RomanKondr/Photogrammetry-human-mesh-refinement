import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from segmentation import segment_human_mesh
from body_parts_allocation import centralize_mesh_upright, group_segments_by_body_parts
from smoothing_hand import remove_humps_and_smooth
from proportions import (
    detect_shoulder_endpoints,
    detect_chest_endpoints,
    detect_waist_endpoints,
    detect_hip_depth_endpoints,
    detect_hip_width_endpoints,
    compute_forward_axis,
    detect_left_leg_deep_endpoints,
)


IN_PATH       = Path(r"C:\Users\alexk\FYP\3D_Model\dense\example.obj")
OUT_CLEANED   = Path(r"C:\Users\alexk\Desktop\body_smoothed.obj")
OUT_ANNOTATED = Path(r"C:\Users\alexk\Desktop\body_proportions.obj")
KEEP_QUANTILE = 0.25


def extract_part_mesh(mesh, vids):
    """
    Extracts the submesh containing only triangles whose vertices are in vids.
    """
    tris = np.asarray(mesh.triangles)
    mask = np.all(np.isin(tris, vids), axis=1)
    keep_vids = np.unique(tris[mask].flatten()).tolist()
    part = mesh.select_by_index(keep_vids, cleanup=True)
    part.compute_vertex_normals()
    return part

def merge_meshes(mesh_list):
    merged = o3d.geometry.TriangleMesh()
    for m in mesh_list:
        merged += m
    merged.remove_duplicated_vertices()
    merged.remove_unreferenced_vertices()
    merged.compute_vertex_normals()
    return merged

def main():
    # 1) load & upright
    mesh = o3d.io.read_triangle_mesh(str(IN_PATH))
    if mesh.is_empty():
        sys.exit("Failed to load input mesh")
    mesh = centralize_mesh_upright(mesh)

    # 2) initial segmentation & grouping
    labels   = segment_human_mesh(mesh, clas=13)
    segments = [np.where(labels == k)[0] for k in range(labels.max()+1)]
    parts    = group_segments_by_body_parts(segments, mesh)

    # 3) smooth each part separately (except head)
    cleaned_parts = []
    for name, seg_ids in parts.items():
        # all vertex indices for this part
        vids = np.unique(np.concatenate([segments[s] for s in seg_ids]))
        part_mesh = extract_part_mesh(mesh, vids)
        if name != 'head':
            # apply hump-removal & smoothing on this isolated part
            part_mesh = remove_humps_and_smooth(
                part_mesh,
                keep_quantile=KEEP_QUANTILE
            )
        # head is left exactly as-is
        cleaned_parts.append(part_mesh)

    # 4) merge them back into one mesh
    cleaned = merge_meshes(cleaned_parts)
    o3d.io.write_triangle_mesh(str(OUT_CLEANED), cleaned)
    print(f"Cleaned body written → {OUT_CLEANED}")

    # 5) re-segment & re-group on cleaned mesh
    labels_c   = segment_human_mesh(cleaned, clas=13)
    segments_c = [np.where(labels_c == k)[0] for k in range(labels_c.max()+1)]
    parts_c    = group_segments_by_body_parts(segments_c, cleaned)

    # 6) detect all the standard markers
    pSL, pSR = detect_shoulder_endpoints(cleaned, parts_c, segments_c)
    pCF, pCB = detect_chest_endpoints   (cleaned, parts_c, segments_c)
    # only if we have ≥2 torso segments
    waist_ok = len(parts_c['torso']) >= 2
    if waist_ok:
        pWL, pWR = detect_waist_endpoints(cleaned, parts_c, segments_c)
    hip_ok = len(parts_c['torso']) >= 2
    if hip_ok:
        pHB, pHF, hip_y = detect_hip_depth_endpoints(cleaned, parts_c, segments_c)
        pHL, pHR       = detect_hip_width_endpoints(cleaned, parts_c, segments_c)
    forward2d = compute_forward_axis(cleaned, parts_c, segments_c)
    pLU, pLBu = detect_left_leg_deep_endpoints(
        cleaned, parts_c, segments_c, which='upper',  forward2d=forward2d
    )
    if len(parts_c['left_leg']) >= 2:
        pLM, pLBm = detect_left_leg_deep_endpoints(
            cleaned, parts_c, segments_c, which='middle', forward2d=forward2d
        )
    else:
        pLM, pLBm = pLU, pLBu

    # 7) collect only valid markers
    markers = {
        'shoulder_width': (pSL, pSR),
        'chest_depth':    (pCF, pCB),
    }
    if waist_ok: markers['waist_width'] = (pWL, pWR)
    if hip_ok:
        markers['hip_depth'] = (pHB, pHF)
        markers['hip_width'] = (pHL, pHR)
    markers['left_leg_upper'] = (pLU, pLBu)
    markers['left_leg_mid']   = (pLM, pLBm)

    # 8) annotate & save
    aabb   = cleaned.get_axis_aligned_bounding_box()
    height = aabb.get_max_bound()[1] - aabb.get_min_bound()[1]
    radius = height * 0.01

    annotated = o3d.geometry.TriangleMesh(cleaned)
    for A, B in markers.values():
        if A is None or B is None:
            continue
        for P in (A, B):
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
            sph.translate(P)
            annotated += sph

    o3d.io.write_triangle_mesh(str(OUT_ANNOTATED), annotated)
    print(f"Annotated proportions mesh written → {OUT_ANNOTATED}")

if __name__ == '__main__':
    main()

