# 18744: Poor Lighting Segmentation
Introduction to Autonomous Driving - Project Topic 7: Computer Vision in Poor Lighting

## Dataset Download

```commandline
mkdir datasets/
cd datasets/
```

### MFNet Dataset
The MFNet dataset must be downloaded from Google Drive.  See the MFNet README for instructions.
Once the zip is in the datasets folder:
```commandline
unzip ir_seg_dataset.zip
```
Then open the folder and change line 5 of the make_flip.py file to:
```python
root_dir = './'
```
And run the flipping program
```commandline
python make_flip.py
```

### HeatNet Dataset
```commandline
mkdir heatnet_data/
cd heatnet_data/
wget http://aisdatasets.informatik.uni-freiburg.de/freiburg-thermal-segmentation/train.zip
wget http://aisdatasets.informatik.uni-freiburg.de/freiburg-thermal-segmentation/test.zip
unzip train.zip
unzip test.zip
```

### Our Dataset
```commandline

```

## How to Run
