# UAV Forest Monitoring Experiments

This repository contains the survey paper, datasets summary, and taxonomy of deep learning methods applied to UAV-based forest monitoring. And reproduction and experimental evaluation of various deep-learning models for UAV-based forest monitoring. Each model is placed in a separate folder, including the original README from its source repository.

## Abstract

Deep learning has proven highly effective in extracting hierarchical features from complex remote sensing data, making it well-suited for forest monitoring. Recent advances with UAV-acquired RGB imagery and LiDAR point clouds have enabled deep neural networks to achieve state-of-the-art performance in tasks such as in- dividual tree detection, species classification, and anomaly detection. This paper surveys deep learning techniques for forest analysis, covering model architectures, loss func- tions, and preprocessing methods tailored to forest data. We review common datasets and evaluation protocols, and assess performance under challenges such as dense canopies, overlapping crowns, and diverse forest structures. Finally, we discuss current limitations, including poor generalization due to dataset diversity, and outline future directions toward more robust, scalable, and input-aware systems for forestry applications.

## Included Models

| Model              | Task                                  | Data  | Notes                                    |
| ------------------ | ------------------------------------- | ----- | ---------------------------------------- |
| [TreeFormer](https://arxiv.org/abs/2307.06118)      | Tree counting (density estimation)    | RGB   | Good treatment of dense forests, semi- supervised, no need for BBox. No sup- port for species classification or anoma- lies      |
| [ATFENet](https://doi.org/10.3390/rs14164113)            | Tree segmentation and counting        | RGB   | Lightweight, real-time running on UAV, integrated photo stitching pipeline. Re- duced performance in complex canopy forests (Acacia)         |
| [YOLOv5 (CHM-based)](https://doi.org/10.1016/j.ophoto.2023.100045) | Tree detection                        | CHM   | Fast, suitable for real-time UAV deploy- ment, taking advantage of tree canopy height. Loss of 3D detail, easy to distort on steep terrain        |
| [ForAINet](https://www.sciencedirect.com/science/article/pii/S0034425724000890)           | 3D semantic and instance segmentation | LiDAR | Keep 3D information intact, accu- rately measure plant biological at- tributes. Consuming computational re- sources, requiring large 3D label data |
| [Point-wise Net](https://doi.org/10.1016/j.engappai.2021.104172)     | Point-supervised segmentation         | RGB   | Reduce labeling costs, match big data. Difficult to separate overlapping trees, low border accuracy                      |


## Datasets
The following datasets were used in the experiments:
- FOR-Instance (LiDAR) - [Paper](https://arxiv.org/abs/2309.01279)/ [Source](https://zenodo.org/records/8287792)
- Acacia (RGB)
- OilPalm (RGB)
- KCL-London (RGB) — [Source](https://drive.google.com/file/d/1xcjv8967VvvzcDM4aqAi7Corkb11T0i2/view)
- Yosemite - [Source](https://drive.google.com/drive/folders/1NWAqslICPoTS8OvT8zosI0R7cmsl6x9j)
- Jiangsu (RGB)

## Taxonomy of Approches

We categorize deep learning methods into three tasks:

- **1. Individual Tree Detection:** 
    - Object detection models used to locate individual tree crowns in UAV imagery.
    - Common challenges: overlapping canopies, variable illumination, dense forests, and scale variation across UAV altitudes.  
- **2. Tree Species Classification:**
    - CNN and Transformer architectures classify tree species from canopy patches or hyperspectral/multispectral imagery.
    - Often requires domain generalization methods due to species appearance changes across regions, seasons, and sensor conditions.
- **3. Forest Anomaly Detection: **
    - Detects disease, defoliation, forest fire damage, or illegal logging using RGB, thermal, or multispectral data.  
    - Typically constrained by limited anomalous samples → semi-supervised and one-class methods are common.

And three data modalities:
- RGB imagery:
  - Most widely used due to low cost and accessibility.
  - Supports detection, segmentation, and basic classification tasks.
  - Limitations: lacks structural depth → limited accuracy for tree height or biomass estimation.
- LiDAR point clouds:
  - Provides accurate canopy height models (CHM) and 3D structure for tree segmentation and volume estimation.
  - Requires specialized point-based neural networks (PointNet, ForAINet) to process irregular 3D data.
  - High cost and limited availability restrict dataset scale.
- Multimodal fusion

## Network Architectures

- 2D CNN-based:
  - YOLOv5: fast realtime detection for tree counting.
  - Faster R-CNN: higher accuracy.
  - U-Net: widely used for canopy/crown segmentation.
- Transformer-based:
  - Global attention enables long-range canopy structure understanding.
  - TreeFormer
- 3D Point-based:
  - Operate on LiDAR point clouds or SfM-generated point sets.
  - ForAINet: tailored for forest point cloud annotation and inventory tasks.
  - Models capture 3D geometry → suitable for height/volume estimation.
- Attention-based Lightweight Models:
  - Optimized for onboard UAV inference with limited computation.
  - ATFENet:
  - Essential for field deployment and real-time monitoring.

## Training Strategies

- Data augmentation (2D & 3D):
  - Rotation, flipping, color jitter, illumination simulation for RGB.
  - Random downsampling, noise injection, point jittering for LiDAR.  
- Semi-supervised learning:
  - Pseudo-labeling on large unlabeled UAV datasets.
  - Consistency regularization for spectral and spatial invariance.
  - Key when annotation cost is prohibitive.
- Domain adaptation:
  - Important for cross-region generalization (e.g., tropical vs temperate forests).
  - Techniques include adversarial domain alignment and feature normalization.
  - Helps models trained in one forest type adapt to another.
- UAV mosaicking pipeline:
  - Orthomosaic creation through Structure-from-Motion (SfM).
  - Produces DSM/DTM, canopy height models, and stitched imagery.
  - Critical preprocessing step for both RGB and LiDAR workflows.


## Result
