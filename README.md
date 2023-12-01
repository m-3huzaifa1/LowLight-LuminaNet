#   Lumina_Net
The implementaion is taken from Zero-DCE++. It is the improved version of Zero_DCE++ with relatively high psnr on low training parameters.

# Pytorch
Pytorch implementation of Low Light LuminaNet: An Enhanced Version of Zero_DCE++

## Requirements
1. Python 3.7 
2. Pytorch 1.0.0
3. opencv
4. torchvision 0.2.1
5. cuda 10.0

LuminaNet does not need special configurations. Just basic environment. 

Or you can create a conda environment to run our code like this:
conda create --name my_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch

### Folder structure
Download the LuminaNet first.
The following shows the basic folder structure.
```

├── data
│   ├── test_data 
│   └── train_data 
├── Test.py # testing code
├── Train.py # training code
├── model.py # Zero-DEC++ network
├── dataloader.py
├── snapshots_LN
```
### Test: 

```
python Test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result_LN" in the "data". You can find the enhanced images in the "result" folder.

### Train:
```
python Train.py 
```
## Bibtex

```
@inproceedings{LuminaNet,
 author = {Huzaifa,Pankaj},
 title = {Low-Light-LuminaNet using Deep Curve Estimation},
 pages    = {},
 month = {},
 year = {2023}
}
```
