# BAFM-and-MAL
Official PyTorch implementation for the following paper:

**Fine-grained Semantic Information Preservation and Misclassification-aware Loss for 3D Point Cloud**

*by Yanmei Zou, Xuefei Lin, [Hongshan Yu](http://eeit.hnu.edu.cn/info/1289/4535.htm)*, [Zhengeng Yang](https://gsy.hunnu.edu.cn/info/1071/3537.htm), [Naveed Akhtar](https://findanexpert.unimelb.edu.au/profile/1050019-naveed-akhtar)

## Features
In the project, we propose a new and flexible codebase for point-based methods, namely [**OpenPoints**](https://github.com/guochengqian/openpoints). The biggest difference between OpenPoints and other libraries is that we focus more on reproducibility and fair benchmarking. 

1. We propose a plug-and-play bilateral attention fusion module (BAFM) to improve the performance of “Encoder-Decoder” structures by preserving fine-grained semantic information for dense multi-classification tasks.

2. We propose a misclassification-aware loss (MAL) function to fully exploit the predicted information by applying a more informed penalty on the misclassified classes.

3. We extend MAL as a novel post-processing operation to improve the performance of existing techniques.



## Installation

```
git clone git@github.com:guochengqian/PointNeXt.git
cd BAFM-and-MAL
source install.sh
```
Note:  

1) the `install.sh` requires CUDA 11.1; if another version of CUDA is used,  `install.sh` has to be modified accordingly; check your CUDA version by: `nvcc --version` before using the bash file;
2) you might need to read the `install.rst` for a step-by-step installation if the bash file (`install.sh`) does not work for you by any chance;
3) for all experiments, we use wandb for online logging by default. Run `wandb --login` only at the first time in a new machine, or set `wandn.use_wandb=False` if you do not want to use wandb. Read the [official wandb documentation](https://docs.wandb.ai/quickstart) if needed.



## Usage 

**Check `README.md` file under `cfgs` directory for detailed training and evaluation on each benchmark.**  

For example, 
* Train and validate on ScanObjectNN for 3D object classification, check [`cfgs/scanobjectnn/README.md`](cfgs/scanobjectnn/README.md)
* Train and validate on S3DIS for 3D segmentation, check [`cfgs/s3dis/README.md`](cfgs/s3dis/README.md)

Note:  
1. We use *yaml* to support training and validation using different models on different datasets. Just use `.yaml` file accordingly. For example, train on ScanObjectNN using PointNeXt: `CUDA_VISIBLE_DEVICES=1 bash script/main_classification.sh cfgs/scanobjectnn/pointnext-s.yaml`, train on S3DIS using ASSANet-L: `CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/assanet-l.yaml`.  
2. Check the default arguments of each .yaml file. You can overwrite them simply through the command line. E.g. overwrite the batch size, just appending `batch_size=32` or `--batch_size 32`.  


## Model Zoo

We provide the **training logs & pretrained models** in column `our released`  *trained with the improved training strategies proposed by our PointNeXt* through Google Drive. 

*TP*: Throughput (instance per second) measured using an NVIDIA Tesla V100 32GB GPU and a 32 core Intel Xeon @ 2.80GHz CPU.

### ScanObjectNN (Hardest variant) Classification

Throughput is measured with 128 x 1024 points. 

| name | OA/mAcc (Original) |OA/mAcc (our released) 
|:---:|:---:|:---:|:---:| :---:|:---:|
|  PointNet   | 68.2 / 63.4 | [75.2 / 71.4](https://drive.google.com/drive/folders/1F9sReTX9MC1RAEHZaSh6o_tn9hgQFT95?usp=sharing)
| DGCNN | 78.1 / 73.6 | [86.1 / 84.3](https://drive.google.com/drive/folders/1KWfvYPrJNdOaMOxTQnwI0eXTspKHCfYQ?usp=sharing) 
| PointMLP |85.4±1.3 / 83.9±1.5 | [87.7 / 86.4](https://drive.google.com/drive/folders/1Cy4tC5YmlbiDATWEW3qLLNxnBx2-XMqa?usp=sharing)
| PointNet++ | 77.9 / 75.4 | [86.2 / 84.4](https://drive.google.com/drive/folders/1T7uvQW4cLp65DnaEWH9eREH4XKTKnmks?usp=sharing)
| **PointNeXt-S** |87.7±0.4 / 85.8±0.6 | [88.20 / 86.84](https://drive.google.com/drive/folders/1A584C9x5uAqppbjNNiVqlA_7uOOOlEII?usp=sharing)|




### S3DIS (Area 5) Segmentation

Throughput (TP) is measured with 16 x 15000 points.

|       name       |    mIoU/OA/mAcc (Original)     |                 mIoU/OA/mAcc (our released)                  
| :--------------: | :----------------------------: | :----------------------------------------------------------: 
|    PointNet++    |        53.5 / 83.0 / -         |                    [63.6 / 88.3 / 70.2](https://drive.google.com/drive/folders/1NCy1Av1-TSs_46ngOk181A3BUhc8hpWV?usp=sharing)                   
| ASSANet | 63.0 / - /- | [65.8 / 88.9 / 72.2](https://drive.google.com/drive/folders/1a-2yNP_JvOgKPTLBTYXP5NtUmspc1P-c?usp=sharing)
| ASSANet-L | 66.8 / - / - | [68.0 / 89.7/ 74.3](https://drive.google.com/drive/folders/1FinOKtFEigsbgjsLybhpZr2xkESLIDhf?usp=sharing) 
| **PointNeXt-S**  | 63.4±0.8 / 87.9±0.3 / 70.0±0.7 |                    [64.2 / 88.2 / 70.7](https://drive.google.com/drive/folders/1UG8hh_CrUf-OhrYbcDd0zvDtoInrGP1u?usp=sharing)               
| **PointNeXt-B**  | 67.3±0.2 / 89.4±0.1 / 73.7±0.6 |                    [67.5 / 89.4 / 73.9](https://drive.google.com/drive/folders/166g_4vaCrS6CSmp3FwAWxl8N8ZmMuylw?usp=sharing)              
| **PointNeXt-L**  | 69.0±0.5 / 90.0±0.1 / 75.3±0.8 |                    [69.3 / 90.1 / 75.7](https://drive.google.com/drive/folders/1g4qE6g10zoZY5y6LPDQ5g12DvSLbnCnj?usp=sharing)                   
| **PointNeXt-XL** | 70.5±0.3 / 90.6±0.2 / 76.8±0.7 | [71.1 / 91.0 / 77.2](https://drive.google.com/drive/folders/1rng7YmfzzIGtXREn7jW0vVFmSSakLQs4?usp=sharing) 





### ModelNet40 Classificaiton



| name | OA/mAcc (Original) |OA/mAcc (our released) | #params | FLOPs | Throughput (ins./sec.) |
|:---:|:---:|:---:|:---:| :---:|:---:|
| PointNet++ | 91.9 / - | [93.0 / 90.7](https://drive.google.com/drive/folders/1Re2_NCtZBKxIhtv755LlnHjz-FBPWjgW?usp=sharing) | 1.5M | 1.7G | 1872 |
| **PointNeXt-S** (C=64) | 93.7±0.3 / 90.9±0.5 | [94.0 / 91.1](https://drive.google.com/drive/folders/14biOHuvH8b2F03ZozrWyF45tCmtsorYN?usp=sharing) | 4.5M | 6.5G | 2033 |

