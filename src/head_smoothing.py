#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import numpy as np
import open3d as o3d
import cv2
import mediapipe as mp
from scipy.spatial import cKDTree

# ───────── User parameters ─────────
MESH_IN        = Path(r"C:\Users\alexk\Desktop\heead_smoothing\head_only.ply")
OUT_IMG        = Path(r"C:\Users\alexk\Desktop\head_front.png")
OUT_2DPTS      = Path(r"C:\Users\alexk\Desktop\mp_facemesh.npy")
OUT_JSON       = Path(r"C:\Users\alexk\Desktop\landmarks.json")
OUT_ANNOT      = Path(r"C:\Users\alexk\Desktop\mp_facemesh_annot.png")
OUT_DEBUG_PLY  = Path(r"C:\Users\alexk\Desktop\head_pts_marked.ply")
OUT_FINAL_PLY  = Path(r"C:\Users\alexk\Desktop\head_smoothed_holefilled.ply")

IMG_W, IMG_H = 640, 640
FX = FY      = 800.0
CX, CY      = IMG_W/2, IMG_H/2

FEATURE_SETS = (
    mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
    mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
    mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
    mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
    mp.solutions.face_mesh.FACEMESH_LIPS,
    mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
    mp.solutions.face_mesh.FACEMESH_NOSE
)

# ───────── Helpers ─────────

def load_and_center(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        sys.exit(f"✗ Could not load mesh: {path}")
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    return mesh

def render_offscreen(mesh: o3d.geometry.TriangleMesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=IMG_W, height=IMG_H, visible=False)
    vis.add_geometry(mesh)

    intr = o3d.camera.PinholeCameraIntrinsic(IMG_W, IMG_H, FX, FY, CX, CY)
    ctr  = vis.get_view_control()
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intr
    params.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    ctr.set_front ((0.0, 0.0, -1.0))
    ctr.set_lookat((0.0, 0.0,  0.0))
    ctr.set_up     ((0.0, 1.0,  0.0))
    ctr.set_zoom   (1.0)

    vis.poll_events(); vis.update_renderer()
    live = ctr.convert_to_pinhole_camera_parameters()
    color_f32 = np.asarray(vis.capture_screen_float_buffer(True))
    depth      = np.asarray(vis.capture_depth_float_buffer(True))
    vis.destroy_window()

    color_u8 = (np.clip(color_f32,0,1)*255).astype(np.uint8)[..., ::-1]
    return color_u8, depth, live.intrinsic, live.extrinsic

def remove_black_holes(color_u8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(color_u8, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.inpaint(color_u8, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def preprocess_for_facemesh(img_bgr: np.ndarray) -> np.ndarray:
    img = remove_black_holes(img_bgr)
    img = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[...,2] = cv2.equalizeHist(hsv[...,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def detect_and_filter(img_bgr: np.ndarray) -> np.ndarray:
    with mp.solutions.face_mesh.FaceMesh(
          static_image_mode=True,
          max_num_faces=1,
          refine_landmarks=True,
          min_detection_confidence=0.5
    ) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        sys.exit("✗ No face detected")
    lm = res.multi_face_landmarks[0].landmark

    h,w,_ = img_bgr.shape
    pts = np.array([[int(p.x*w), int(p.y*h)] for p in lm], dtype=int)

    keep = set()
    for conn in FEATURE_SETS:
        for i,j in conn:
            keep.add(i); keep.add(j)
    return pts[np.array(sorted(keep),dtype=int)]

def backproject(pts2d, depth, intr, extr):
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()
    inv = np.linalg.inv(extr)
    H,W = depth.shape
    out = []
    for u,v in pts2d:
        ui,vi = int(round(u)), int(round(v))
        if not (0<=ui<W and 0<=vi<H): continue
        z = depth[vi,ui]
        if z<=0: continue
        x = (u-cx)*z/fx
        y = (v-cy)*z/fy
        p_cam = np.array([x,y,z,1.0])
        out.append((inv @ p_cam)[:3])
    return np.vstack(out)

def sprinkle_spheres(mesh, pts3d, out_ply):
    mesh.compute_vertex_normals()
    r = max(mesh.get_max_bound() - mesh.get_min_bound()) * 0.01
    debug = o3d.geometry.TriangleMesh(mesh)
    for p in pts3d:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=r,resolution=12)
        sph.paint_uniform_color([1,0,0])
        sph.translate(p)
        debug += sph
    o3d.io.write_triangle_mesh(str(out_ply), debug)

def laplacian_preserve(mesh, anchor_idx, iterations=10):
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    C = np.asarray(mesh.vertex_colors)
    adj = [[] for _ in range(len(V))]
    for i,j,k in F:
        adj[i]+= [j,k]; adj[j]+= [i,k]; adj[k]+= [i,j]
    adj = [list(set(n)) for n in adj]

    anchors = set(anchor_idx)
    Vn = V.copy()
    for _ in range(iterations):
        Vo = Vn.copy()
        for vid in range(len(V)):
            if vid in anchors: continue
            nbr = adj[vid]
            if nbr:
                Vn[vid] = Vo[nbr].mean(0)

    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(Vn),
        o3d.utility.Vector3iVector(F))
    out.vertex_colors = o3d.utility.Vector3dVector(C)
    out.compute_vertex_normals()
    return out

def poisson_reconstruct_with_color(mesh, depth=8):
    pcd = o3d.geometry.PointCloud()
    pcd.points  = mesh.vertices
    pcd.normals = mesh.vertex_normals
    poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1)
    bbox = mesh.get_axis_aligned_bounding_box()
    crop = poisson.crop(bbox)
    # color transfer
    orig_pts  = np.asarray(mesh.vertices)
    orig_cols = np.asarray(mesh.vertex_colors)
    new_pts   = np.asarray(crop.vertices)
    tree      = cKDTree(orig_pts)
    _, idx    = tree.query(new_pts)
    crop.vertex_colors = o3d.utility.Vector3dVector(orig_cols[idx])
    crop.compute_vertex_normals()
    return crop

# ───────── Main ─────────

def main():
    mesh = load_and_center(MESH_IN)

    # 1) screenshot + depth
    color, depth, intr, extr = render_offscreen(mesh)
    cv2.imwrite(str(OUT_IMG), color)
    print("▶ Screenshot →", OUT_IMG)

    # 2) preprocess & initial detect
    pp   = preprocess_for_facemesh(color)
    rough = detect_and_filter(pp)

    # 3) crop to face region + upscale
    pad = 20
    x0 = max(0, rough[:,0].min()-pad)
    x1 = min(IMG_W, rough[:,0].max()+pad)
    y0 = max(0, rough[:,1].min()-pad)
    y1 = min(IMG_H, rough[:,1].max()+pad)
    crop = pp[y0:y1, x0:x1]
    up = cv2.resize(crop, (512,512))

    # 4) high-res detect → map back
    fine = detect_and_filter(up)
    sx, sy = (x1-x0)/512.0, (y1-y0)/512.0
    pts2d = np.stack([
        fine[:,0]*sx + x0,
        fine[:,1]*sy + y0
    ], axis=-1)

    # 5) save 2D & JSON
    np.save(str(OUT_2DPTS), pts2d)
    lm_json = {
      "intrinsic": intr.intrinsic_matrix.tolist(),
      "extrinsic": extr.tolist(),
      "landmarks": [
        {"pixel": [float(u),float(v)]}
        for u,v in pts2d
      ]
    }
    with open(OUT_JSON, "w") as f:
        json.dump(lm_json, f, indent=2)
    print("▶ 2D pts →", OUT_2DPTS, "and", OUT_JSON)

    # 6) annotate & save
    ann = color.copy()
    for (u,v) in pts2d.astype(int):
        cv2.circle(ann, (u,v), 2, (0,255,0), -1)
    cv2.imwrite(str(OUT_ANNOT), ann)
    print("▶ Annotated →", OUT_ANNOT)

    # 7) back-project & sprinkle
    pts3d = backproject(pts2d, depth, intr, extr)
    sprinkle_spheres(mesh, pts3d, OUT_DEBUG_PLY)
    print("▶ Debug mesh →", OUT_DEBUG_PLY)

    # 8) anchors & smooth+hole-fill
    tree = o3d.geometry.KDTreeFlann(mesh)
    anchors = []
    for p in pts3d:
        _, idx, _ = tree.search_knn_vector_3d(p,1)
        anchors.append(idx[0])
    anchors = np.unique(anchors)

    smooth = laplacian_preserve(mesh, anchors, iterations=20)
    final  = poisson_reconstruct_with_color(smooth, depth=8)
    o3d.io.write_triangle_mesh(str(OUT_FINAL_PLY), final)
    print("✅ Final mesh →", OUT_FINAL_PLY)

if __name__=="__main__":
    main()
