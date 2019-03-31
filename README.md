# POCONet
Road Way to Safety

CNN Model

We are using YOLOv2 model architecture to detect potholes.
Model is trained on a dataset containing 259 images having a total of 829 potholes in them.

To detect potholes on any image or video just run the following command-
python predict.py -c config.json -w saved_weights.h5(path of the saved weight should be provided) -i (Image or video path)

## Image

![Road with potholes](image/potholes63.jpg?raw=true "Road with Potholes")

## Detected potholes
![Road with potholes detected](image/potholes63_detected.jpg?raw=true "Road with Potholes Detected")


To Visualize detected potholes on google map simply go through maps.ipynb . 
