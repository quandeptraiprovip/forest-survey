# UAV Forest Monitoring Experiments

This repository contains the reproduction and experimental evaluation of various deep-learning models for UAV-based forest monitoring. Each model is placed in a separate folder, including the original README from its source repository.

## Included Models

| Model              | Task                                  | Data  | Notes                                    |
| ------------------ | ------------------------------------- | ----- | ---------------------------------------- |
| TreeFormer         | Tree counting (density estimation)    | RGB   | Good treatment of dense forests, semi- supervised, no need for BBox. No sup- port for species classification or anoma- lies      |
| ATFENet            | Tree segmentation and counting        | RGB   | Lightweight, real-time running on UAV, integrated photo stitching pipeline. Re- duced performance in complex canopy forests (Acacia)         |
| YOLOv5 (CHM-based) | Tree detection                        | CHM   | Fast, suitable for real-time UAV deploy- ment, taking advantage of tree canopy height. Loss of 3D detail, easy to distort on steep terrain        |
| ForAINet           | 3D semantic and instance segmentation | LiDAR | Keep 3D information intact, accu- rately measure plant biological at- tributes. Consuming computational re- sources, requiring large 3D label data |
| Point-wise Net     | Point-supervised segmentation         | RGB   | Reduce labeling costs, match big data. Difficult to separate overlapping trees, low border accuracy                      |


## Datasets
The following datasets were used in the experiments:
- FOR-Instance (LiDAR) - [Paper](https://arxiv.org/abs/2309.01279)/ [Source](https://zenodo.org/records/8287792)
- Acacia (RGB)
- OilPalm (RGB)
- KCL-London (RGB) — [Source](https://drive.google.com/file/d/1xcjv8967VvvzcDM4aqAi7Corkb11T0i2/view)
- Jiangsu (RGB)

## Result
