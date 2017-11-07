# Detecting the presence of clouds using satellite image data

## Task
- Modeling cloud detection in polar regions based on radiances recorded automatically by the MISR sensor aboard the NASA satellite Terr

## Data
- 3 satellite images
- “Expert labels” used for model training for each point in image
- Features
    - NDAI, SD, CORR (features based on subject knowledge, see Yu2008.pdf) 
    - Radiance angles (DF, CF, BF, AF, AN, see http://www-misr.jpl.nasa.gov/) 
   
## Data Processing
- Created 9 cross validation sets by splitting each image into 9 approximately equal sized blocks
- Each training split of a cross validation set contains 7 of 9 blocks of each image
- Each testing split of a cross validation set contains 2 of 9 blocks of each image
- Extracted feature sets of 4 closest neighbors for each point (neighbors)

## Cloud Repo Data
- Cross validation sets contained within Cloud/src/main/resources/
  - Testing splits contained in Cloud/src/main/resources/test
  - Training splits contained in Cloud/src/main/resources/train
- Cross validaiton sets using neighbor data contained within Cloud/src/main/resources/
  - Testing splits contained in Cloud/src/main/resources/neighborTest
  - Training splits contained in Cloud/src/main/resources/neighborTrain
  - n1 contains features and label for a point
  - n2 contains features for the closest neighbor
  - n3 contains features for the 2nd closest neighbor
  - n4 contains features for the 3rd closest neighbor
  - n5 contains features for the 4th closest neighbor
  
  
## Neural Networks
- Cloud.java: Contains code for a regular feed forward network using the 8 features using MultiLayerNetwork
- Cloud_neighbor.java: Contains code for a feed forward of 8 original features + features of closest neighbors using ComputationGraph + mergeVertex

## Results

### Cloud.java:

#### CV1

Examples labeled as 0 classified by model as 0: 1750 times
Examples labeled as 0 classified by model as 1: 5303 times
Examples labeled as 1 classified by model as 0: 295 times
Examples labeled as 1 classified by model as 1: 40997 times


 Classes:    2
 Accuracy:        0.8842
 Precision:       0.8706
 Recall:          0.6205
 F1 Score:        0.9361

FINAL TEST AUC: 0.8551935698373123

#### CV2

Examples labeled as 0 classified by model as 0: 8839 times
Examples labeled as 0 classified by model as 1: 2631 times
Examples labeled as 1 classified by model as 0: 189 times
Examples labeled as 1 classified by model as 1: 36305 times


 Classes:    2
 Accuracy:        0.9412
 Precision:       0.9557
 Recall:          0.8827
 F1 Score:        0.9626

FINAL TEST AUC: 0.9899205604121408

#### CV3

Examples labeled as 0 classified by model as 0: 11531 times
Examples labeled as 0 classified by model as 1: 1393 times
Examples labeled as 1 classified by model as 0: 9428 times
Examples labeled as 1 classified by model as 1: 30296 times


 Classes:    2
 Accuracy:        0.7945
 Precision:       0.7531
 Recall:          0.8274
 F1 Score:        0.8485

FINAL TEST AUC: 0.8944111498662976

#### CV4

Examples labeled as 0 classified by model as 0: 5435 times
Examples labeled as 0 classified by model as 1: 5279 times
Examples labeled as 1 classified by model as 0: 2524 times
Examples labeled as 1 classified by model as 1: 26027 times


 Classes:    2
 Accuracy:        0.8013
 Precision:       0.7571
 Recall:          0.7094
 F1 Score:        0.8696
 
FINAL TEST AUC: 0.8375676141867809


#### CV5

Examples labeled as 0 classified by model as 0: 8519 times
Examples labeled as 0 classified by model as 1: 100 times
Examples labeled as 1 classified by model as 0: 100 times
Examples labeled as 1 classified by model as 1: 12655 times


 Classes:    2
 Accuracy:        0.9906
 Precision:       0.9903
 Recall:          0.9903
 F1 Score:        0.9922

FINAL TEST AUC: 0.9989859017902203


#### CV6


Examples labeled as 0 classified by model as 0: 3735 times
Examples labeled as 0 classified by model as 1: 6979 times
Examples labeled as 1 classified by model as 0: 958 times
Examples labeled as 1 classified by model as 1: 27593 times


 Classes:    2
 Accuracy:        0.7979
 Precision:       0.7970
 Recall:          0.6575
 F1 Score:        0.8743

FINAL TEST AUC: 0.8449579860040033


#### CV7

Examples labeled as 0 classified by model as 0: 9036 times
Examples labeled as 0 classified by model as 1: 3888 times
Examples labeled as 1 classified by model as 0: 4541 times
Examples labeled as 1 classified by model as 1: 35183 times



 Classes:    2
 Accuracy:        0.8399
 Precision:       0.7830
 Recall:          0.7924
 F1 Score:        0.8930

FINAL TEST AUC: 0.886610437148007


#### CV8

Examples labeled as 0 classified by model as 0: 9545 times
Examples labeled as 0 classified by model as 1: 1925 times
Examples labeled as 1 classified by model as 0: 269 times
Examples labeled as 1 classified by model as 1: 36225 times


 Classes:    2
 Accuracy:        0.9543
 Precision:       0.9611
 Recall:          0.9124
 F1 Score:        0.9706

FINAL TEST AUC: 0.9911603036307899

#### CV9

Examples labeled as 0 classified by model as 0: 5785 times
Examples labeled as 0 classified by model as 1: 1268 times
Examples labeled as 1 classified by model as 0: 10635 times
Examples labeled as 1 classified by model as 1: 30657 times


 Classes:    2
 Accuracy:        0.7538
 Precision:       0.6563
 Recall:          0.7813
 F1 Score:        0.8374

FINAL TEST AUC: 0.8869103118033523



### Cloud_neighbor.java:

#### CV1:

Examples labeled as 0 classified by model as 0: 203 times
Examples labeled as 0 classified by model as 1: 6850 times
Examples labeled as 1 classified by model as 0: 97 times
Examples labeled as 1 classified by model as 1: 41195 times


 Classes:    2
 Accuracy:        0.8563
 Precision:       0.7670
 Recall:          0.5132
 F1 Score:        0.9222

FINAL TEST AUC: 0.8376413692956247


#### CV2:

Examples labeled as 0 classified by model as 0: 10189 times
Examples labeled as 0 classified by model as 1: 1281 times
Examples labeled as 1 classified by model as 0: 297 times
Examples labeled as 1 classified by model as 1: 36197 times



 Classes:    2
 Accuracy:        0.9671
 Precision:       0.9687
 Recall:          0.9401
 F1 Score:        0.9787

FINAL TEST AUC: 0.9906514710290261

#### CV3:

Examples labeled as 0 classified by model as 0: 8418 times
Examples labeled as 0 classified by model as 1: 4506 times
Examples labeled as 1 classified by model as 0: 3033 times
Examples labeled as 1 classified by model as 1: 36691 times


 Classes:    2
 Accuracy:        0.8568
 Precision:       0.8129
 Recall:          0.7875
 F1 Score:        0.9068

FINAL TEST AUC: 0.8986892098296227

#### CV4:

Examples labeled as 0 classified by model as 0: 3130 times
Examples labeled as 0 classified by model as 1: 7584 times
Examples labeled as 1 classified by model as 0: 484 times
Examples labeled as 1 classified by model as 1: 28067 times



 Classes:    2
 Accuracy:        0.7945
 Precision:       0.8267
 Recall:          0.6376
 F1 Score:        0.8743

FINAL TEST AUC: 0.8904161829028414

#### CV5:

Examples labeled as 0 classified by model as 0: 8616 times
Examples labeled as 0 classified by model as 1: 3 times
Examples labeled as 1 classified by model as 0: 180 times
Examples labeled as 1 classified by model as 1: 12575 times



 Classes:    2
 Accuracy:        0.9914
 Precision:       0.9896
 Recall:          0.9928
 F1 Score:        0.9928

FINAL TEST AUC: 0.99962095191837

#### CV6:

Examples labeled as 0 classified by model as 0: 5064 times
Examples labeled as 0 classified by model as 1: 5650 times
Examples labeled as 1 classified by model as 0: 593 times
Examples labeled as 1 classified by model as 1: 27958 times



 Classes:    2
 Accuracy:        0.8410
 Precision:       0.8635
 Recall:          0.7259
 F1 Score:        0.8996

FINAL TEST AUC: 0.9145322357578269

#### CV7:

Examples labeled as 0 classified by model as 0: 7988 times
Examples labeled as 0 classified by model as 1: 4936 times
Examples labeled as 1 classified by model as 0: 2723 times
Examples labeled as 1 classified by model as 1: 37001 times



 Classes:    2
 Accuracy:        0.8545
 Precision:       0.8140
 Recall:          0.7748
 F1 Score:        0.9062

FINAL TEST AUC: 0.8832670981417631


#### CV8:

Examples labeled as 0 classified by model as 0: 10095 times
Examples labeled as 0 classified by model as 1: 1375 times
Examples labeled as 1 classified by model as 0: 157 times
Examples labeled as 1 classified by model as 1: 36337 times



 Classes:    2
 Accuracy:        0.9681
 Precision:       0.9741
 Recall:          0.9379
 F1 Score:        0.9794

FINAL TEST AUC: 0.9944596717962488

#### CV9: 

Examples labeled as 0 classified by model as 0: 6357 times
Examples labeled as 0 classified by model as 1: 696 times
Examples labeled as 1 classified by model as 0: 18528 times
Examples labeled as 1 classified by model as 1: 22764 times



 Classes:    2
 Accuracy:        0.6024
 Precision:       0.6129
 Recall:          0.7263
 F1 Score:        0.7031

FINAL TEST AUC: 0.8898892132172168
  
