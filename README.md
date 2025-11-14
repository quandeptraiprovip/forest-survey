# UAV Forest Monitoring Experiments

This repository contains the reproduction and experimental evaluation of various deep-learning models for UAV-based forest monitoring. Each model is placed in a separate folder, including the original README from its source repository.

## Included Models

| Model              | Task                                  | Data  | Notes                                    |
| ------------------ | ------------------------------------- | ----- | ---------------------------------------- |
| TreeFormer         | Tree counting (density estimation)    | RGB   | Strong performance on dense forests      |
| ATFENet            | Tree segmentation and counting        | RGB   | Lightweight and suitable for UAV         |
| YOLOv5 (CHM-based) | Tree detection                        | CHM   | Fast but sensitive to CHM quality        |
| ForAINet           | 3D semantic and instance segmentation | LiDAR | Highest accuracy, requires heavy compute |
| Point-wise Net     | Point-supervised segmentation         | RGB   | Low annotation cost                      |


## Datasets
The following datasets were used in the experiments:
FOR-Instance (LiDAR)
Acacia (RGB)
OilPalm (RGB)
KCL-London (RGB)
Jiangsu (RGB)

## Result
