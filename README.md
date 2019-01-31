# AIFuture

## 1 NNTest
### 1.0 Base Usage
```python
python3 NNTest.py
```
### 1.1 Loss Results
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/NNTest/images/loss.png"/></div>
<div align=center>Fig.1.1 Loss</div>

### 1.2 Train Results
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/NNTest/images/results.png"/></div>
<div align=center>Fig.1.2 Train result</div>

### 1.3 Load Model for Transfer Training
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/NNTest/images/loadModelPredict.png"></div>  
<div align=center>Fig.1.3 Accuracy</div>

## 2 FaceRecognition
### 2.1 Base Usage
```python
python3 findFaceInImages.py
```
### 2.2 Single Face
#### 2.2.1 Source Image
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/FaceRecognition/images/Mac.png"/></div>  
<div align=center>Fig.2.1 Single face</div>

#### 2.2.2 Marked image with bounding box
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/FaceRecognition/processed/1.png"/></div>  
<div align=center>Fig.2.2 Bounding box</div>

### 2.3 Multi face
#### 2.3.1 Source Image
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/FaceRecognition/images/AllStarEast.jpeg"/></div>
<div align=center>Fig.2.3 Multi face</div>

#### 2.3.2 Marked image with bounding boxes
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/FaceRecognition/processed/AllStarEast.png"/></div>
<div align=center>Fig.2.4 Bounding box</div>

## 3 Get tickets info
### 3.1 Base Usage
```python
python3 getData.py 深圳 上海 2018-12-25
```
### 3.2 Result
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/ticketsPy/results/resultsone.png"></div>
<div align=center>Fig.3.1 Search info</div>

## 4 Get face images
### 4.1 Base Usage
```python
python3 getImage.py
```
### 4.2 Get sixty images
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/getFaceImage/faceImages/getSixtyFace.png"></div>
<div align=center>Fig.5.1 Download images</div>

### 4.3 Downloading status
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/getFaceImage/faceImages/downloadStatus.png"></div>
<div align=center>Fig.5.2 Download status</div>

## 5 Cluster Algorithm
### 5.1 Base Usage
```python
python2 kmeans.py
```
### 5.2 Results
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/clusterAlgorithm/results/cluster.png"></div>
<div align=center>Fig.5.1 Cluster</div>

## 6 NN tensorboard display
### 6.1 Base Usage
```python
python3 NNtensorboard.py
tensorboard --logdir=~/logs
```
### 6.2 Reuslts
#### 6.2.1 Global Graph Architecture
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/tensorboard/NN/results/GraphGlobal.png"></div>
<div align=center>Fig.6.1 Global graph</div>

#### 6.2.2 Module Graph Architecture
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/tensorboard/NN/results/GraphModule.png"></div>
<div align=center>Fig.6.2 Module graph</div>

## 7 VGGNet
### 7.1 Base Usage
- run
```python
python3 VGG_Cifar.py
```
- baseboard
```python
tensorboard --logdir=/logdir
```

### 7.2 VGGNet Structure
<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/VGGNet/train_result_image/vgg_cifar.png"></div>
<div align=center>Fig.7.1 VGGNet sturcture</div>


### 7.2 Results

<div align="center"><img src="https://github.com/xindaqi/AIFuture/blob/master/VGGNet/train_result_image/sourcedata.png"></div>
<div align="center">Fig.7.2 Source Data</div>

<div align="center" src="https://github.com/xindaqi/AIFuture/blob/master/VGGNet/train_result_image/conv_gp_1.png"></div>
<div align="center">Fig.7.3 Different convlution image feature</div>

<div align=center><img src="https://github.com/xindaqi/AIFuture/blob/master/VGGNet/train_result_image/cost.png"></div>
<div align=center>Fig.7.4 Cross entropy</div>




 

