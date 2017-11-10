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


  
