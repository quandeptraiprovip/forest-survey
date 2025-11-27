# UAV Forest Monitoring Experiments

This repository contains the survey paper, datasets summary, and taxonomy of deep learning methods applied to UAV-based forest monitoring. And reproduction and experimental evaluation of various deep-learning models for UAV-based forest monitoring. Each model is placed in a separate folder, including the original README from its source repository.

## Abstract

Deep learning has proven highly effective in extracting hierarchical features from complex remote sensing data, making it well-suited for forest monitoring. Recent advances with UAV-acquired RGB imagery and LiDAR point clouds have enabled deep neural networks to achieve state-of-the-art performance in tasks such as in- dividual tree detection, species classification, and anomaly detection. This paper surveys deep learning techniques for forest analysis, covering model architectures, loss func- tions, and preprocessing methods tailored to forest data. We review common datasets and evaluation protocols, and assess performance under challenges such as dense canopies, overlapping crowns, and diverse forest structures. Finally, we discuss current limitations, including poor generalization due to dataset diversity, and outline future directions toward more robust, scalable, and input-aware systems for forestry applications.

## Included Models

| Model              | Task                                  | Data  | Notes                                    |
| ------------------ | ------------------------------------- | ----- | ---------------------------------------- |
| [TreeFormer](https://arxiv.org/abs/2307.06118)      | Tree counting (density estimation)    | RGB   | Good treatment of dense forests, semi- supervised, no need for BBox. No sup- port for species classification or anoma- lies      |
| ATFENet            | Tree segmentation and counting        | RGB   | Lightweight, real-time running on UAV, integrated photo stitching pipeline. Re- duced performance in complex canopy forests (Acacia)         |
| YOLOv5 (CHM-based) | Tree detection                        | CHM   | Fast, suitable for real-time UAV deploy- ment, taking advantage of tree canopy height. Loss of 3D detail, easy to distort on steep terrain        |
| [ForAINet](https://www.sciencedirect.com/science/article/pii/S0034425724000890)           | 3D semantic and instance segmentation | LiDAR | Keep 3D information intact, accu- rately measure plant biological at- tributes. Consuming computational re- sources, requiring large 3D label data |
| Point-wise Net     | Point-supervised segmentation         | RGB   | Reduce labeling costs, match big data. Difficult to separate overlapping trees, low border accuracy                      |


## Datasets
The following datasets were used in the experiments:
- FOR-Instance (LiDAR) - [Paper](https://arxiv.org/abs/2309.01279)/ [Source](https://zenodo.org/records/8287792)
- Acacia (RGB)
- OilPalm (RGB)
- KCL-London (RGB) — [Source](https://drive.google.com/file/d/1xcjv8967VvvzcDM4aqAi7Corkb11T0i2/view)
- Jiangsu (RGB)

## Taxonomy of Approches

We categorize deep learning methods into three tasks:
1. Individual Tree Detection  
2. Tree Species Classification  
3. Forest Anomaly Detection  

And three data modalities:
- RGB imagery
- LiDAR point clouds
- Multimodal fusion

## Network Architectures

- 2D CNN-based: YOLOv5, Faster R-CNN
- Transformer-based: TreeFormer
- 3D Point-based: ForAINet
- Attention-based Lightweight Models: ATFENet

## Training Strategies

- Data augmentation (2D & 3D)
- Semi-supervised learning
- Domain adaptation
- UAV mosaicking pipeline


## Result
