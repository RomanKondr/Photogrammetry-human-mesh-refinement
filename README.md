# Photogrammetry Human Mesh Refinement

Automated Python pipeline for refining noisy photogrammetry-derived 3D human meshes.

## Overview

Photogrammetry is a low-cost way to reconstruct 3D human models from photographs, but the resulting meshes often contain surface noise, anatomical distortion, and missing detail. This project presents a Python-based post-processing pipeline that improves noisy photogrammetry-generated meshes through segmentation, region-aware smoothing, facial landmark preservation, and anthropometric measurement extraction. :contentReference[oaicite:0]{index=0}

The goal of the project is to provide an accessible alternative to expensive 3D scanning systems and time-consuming manual cleanup workflows, while still preserving important body structure and facial identity. 

## How It Works

The pipeline is organised into several stages:

### 1. Mesh segmentation
The input mesh is segmented into up to 13 anatomical regions using saliency-based analysis, Laplacian eigen-decomposition, and clustering. These segments are then assigned to body parts such as the torso, arms, legs, and head using spatial heuristics. :contentReference[oaicite:2]{index=2}

### 2. Region-aware smoothing
Instead of applying one global smoothing operation to the whole body, each body region is processed independently. This helps remove noise while avoiding unnecessary distortion of important anatomical structure. Region-specific Laplacian smoothing is used to preserve body proportions more effectively than generic full-mesh smoothing. :contentReference[oaicite:3]{index=3}

### 3. Facial landmark locking
For the head region, 2D facial landmarks are detected with MediaPipe from a front-facing image. These landmarks are back-projected into 3D and used as anchor points during smoothing, so key facial features such as the eyes, nose, lips, and jawline remain stable. :contentReference[oaicite:4]{index=4}

### 4. Proportion measurement
After refinement, the system extracts anthropometric body measurements such as chest width, waist width, hip depth, and related proportions using vertical slicing and bounding-box-based geometric rules. :contentReference[oaicite:5]{index=5}

## Results

The developed prototype successfully processed photogrammetry-derived 3D human meshes through segmentation, region-aware smoothing, and anthropometric measurement extraction. 

Key outcomes include:

- consistent segmentation into anatomical regions across test meshes :contentReference[oaicite:7]{index=7}
- smoothing that removes photogrammetric surface defects while preserving important structure :contentReference[oaicite:8]{index=8}
- landmark-guided head smoothing that preserves facial detail better than general smoothing without landmark locking :contentReference[oaicite:9]{index=9}
- repeatable body measurement outputs aligned with expected spatial structure of the mesh :contentReference[oaicite:10]{index=10}

The results show that the pipeline can improve mesh clarity and usability while remaining transparent, modular, and cost-effective. :contentReference[oaicite:11]{index=11}

## Example Outputs

### Segmentation
![Segmented mesh output](images/segmentation-output.png)

### Before and after smoothing
![Before and after refinement](images/before-after-smoothing.png)

### Head smoothing with landmark preservation
![Head smoothing comparison](images/head-landmark-locking.png)

### Body proportion detection
![Body proportion extraction](images/proportion-detection.png)

## Project Structure

```text
src/
  body_parts_allocation.py
  head_smoothing.py
  proportions.py
  segmentation.py
  smoothing_hand.py
  smoothing_model.py
requirements.txt
