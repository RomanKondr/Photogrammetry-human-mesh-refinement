# Photogrammetry Human Mesh Refinement

Automated Python pipeline for refining noisy photogrammetry-derived 3D human meshes.

## Overview

Photogrammetry is a low-cost way to reconstruct 3D human models from photographs, but when images are captured without professional scanners, the resulting meshes are often noisy, incomplete, and anatomically distorted. In many cases, these models require manual cleanup and smoothing in third-party software such as Blender or MeshLab, which can be time-consuming and usually requires prior experience.

This project explores an automated Python pipeline for cleaning and smoothing noisy photogrammetry-derived 3D human meshes, with the aim of reducing manual post-processing and making refinement more accessible for non-expert users.

## Current Status

The project was developed incrementally, starting from partial meshes before moving toward full-body processing.

At its current stage, the strongest results have been achieved on:

- isolated arm smoothing
- head smoothing
- facial landmark preservation
- full-body segmentation into anatomical regions
- anthropometric marker and proportion extraction

The full end-to-end pipeline is still incomplete. In particular:

- torso and leg smoothing still require further testing
- body-part-specific smoothing logic still needs refinement
- merging individually smoothed parts back into one final refined full-body mesh has not yet been completed

## How It Works

The pipeline is organised into several stages:

### 1. Mesh segmentation
The input mesh is segmented into up to 13 anatomical regions using saliency-based analysis, Laplacian eigen-decomposition, and clustering. These segments are then assigned to body parts such as the torso, arms, legs, and head using spatial heuristics.

### 2. Region-aware smoothing
Instead of applying one global smoothing operation to the whole body, each body region is processed independently. This helps remove noise while avoiding unnecessary distortion of important anatomical structure.

### 3. Facial landmark locking
For the head region, 2D facial landmarks are detected with MediaPipe from a front-facing image. These landmarks are back-projected into 3D and used as anchor points during smoothing so that key facial features such as the eyes, nose, lips, and jawline remain stable.

### 4. Proportion measurement
After refinement, the system extracts anthropometric body measurements such as shoulder width, chest depth, waist width, hip depth, and limb length using geometric rules based on detected body markers.

## Results

The current prototype demonstrates:

- successful smoothing of noisy arm meshes
- head refinement with facial landmark preservation
- segmentation of full-body meshes into anatomical regions
- body-part allocation from segmented clusters
- extraction of body markers, distances, and limb-length ratios

These results suggest that the pipeline can improve noisy photogrammetry outputs while remaining transparent, modular, and accessible. However, full-body smoothing and reconstruction into a single final refined mesh remain future work.

## Example Outputs

### Arm smoothing

| Raw noisy arm mesh | Smoothed arm mesh |
|---|---|
| ![Raw arm mesh](<images/Screenshot 2026-04-03 213210.png>) | ![Smoothed arm mesh](<images/Screenshot 2026-04-03 213217.png>) |

### Surface saliency / analysis on arm mesh

![Arm saliency visualisation](<images/Screenshot 2026-04-03 213733.png>)

### Head refinement

| Original head mesh | Smoothed head mesh |
|---|---|
| ![Original head mesh](<images/Screenshot 2026-04-03 213237.png>) | ![Smoothed head mesh](<images/Screenshot 2026-04-03 213300.png>) |

### Full-body segmentation examples

| Segmentation example 1 | Segmentation example 2 |
|---|---|
| ![Segmentation example 1](<images/Screenshot 2026-04-03 213308.png>) | ![Segmentation example 2](<images/Screenshot 2026-04-03 213334.png>) |

### Additional segmentation example

![Additional segmentation example](<images/Screenshot 2026-04-03 213342.png>)

### Body-part allocation output

![Body-part allocation output](<images/Screenshot 2026-04-03 213326.png>)

### Marker detection on full body

![Marker detection on full body](<images/Screenshot 2026-04-03 213347.png>)

### Extracted body measurements

![Extracted body measurements](<images/Screenshot 2026-04-03 213138.png>)

## Project Structure

```text
src/
  body_parts_allocation.py
  head_smoothing.py
  proportions.py
  segmentation.py
  smoothing_hand.py
  smoothing_model.py
images/
requirements.txt
