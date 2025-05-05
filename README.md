# Central_Line_Challenge
For those using the School of Computing GPU Server, you may skip directly to training the networks step. The required anaconda environment has already been created for you, and the dataset has been pre-downloaded and prepared for you in advance. The following instructions explain how to reproduce the setup on a local computer.

## Create conda environment    

For those wishing to run on a local computer:  
1. Ensure Anaconda has been installed on your device: https://www.anaconda.com/products/distribution  
> - during installation, make sure to select the option to add anaconda to your search path  
2. Create a new conda environment
```
conda create -n createPytorchEnv python=3.9
```  
3. Follow the step-by-step instructions provided [here](https://pytorch.org/get-started/locally/) for installing pytorch with GPU support if necessary. For those using the School of Computing resources, follow the instructions for a linux installation.  
4. Install the remaining requirements  
```
conda activate createPytorchEnv  
pip install ultralytics, pandas, scikit-learn, matplotlib, opencv-python
```
## Clone this repository
1. Using terminal navigate to the directory where you would like to place your local copy of the repository.  
   E.g.:
```
cd C:/Users/SampleUser/Documents
```
2. Clone the repository using git
```
git clone https://github.com/CREATE-Table-Top-Challenge/Central_Line_Challenge.git
```
## Download Data
Download links are password protected and will only be available until May 7th, 2025. Registered participants will receive the password via email on April 16th, 2025.  
  
#### Training Data:
Training data can be downloaded in 10 parts using the following links: [Part 1](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/ESbMYjKQ3g9NqkXg1B7QgjwBoOOirywgPfn-tOffpMKBLA?download=1), [Part 2](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/EaTlGgjtmtlMpq64T72h3hsBkjJ3aofNFLwo8F7Sw84D5Q?download=1), [Part 3](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/EcVmL9Ddj1BBqlkc6OdpVRUBs7h65meOe6i7NEDXoHM3IA?download=1), [Part 4](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/EYrvRPidMz9Fonuh_ljoAtABFtuJVBWb64hI_EWYM2Me-A?download=1), [Part 5](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/ETRNzC7H7rNGtBvmT7kXKWIBrXgT6uZNv7LqPA-wdEzshw?download=1), [Part 6](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/ETPoZIVbgvZHsMJUZ_uUWRoB7vPmqZEodh-gHBOR1jWiLQ?download=1), [Part 7](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/EQ0YscnYz2dIvtDgr50L0boBqBDMdyTzE9S39vGb1xxs0Q?download=1), [Part 8](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/EdwCfiW4fx1Ch0iGcpaJ5HUBJA6zepMQOKRNy99_LGN1gQ?download=1), [Part 9](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/ERMUZvNkHh1Grkb-nADYcJcBDLtUxYrtsnIO_HNv2x2IsA?download=1), [Part 10](https://queensuca-my.sharepoint.com/:u:/g/personal/11rjh10_queensu_ca/EXwElhZDjotDhMCs7E68nW8Btm2VsC8Ap9PheAKYtBgHxw?download=1)  

#### Unlabelled Data:
Unlabelled data can be found [here](https://queensuca-my.sharepoint.com/:f:/g/personal/11rjh10_queensu_ca/EmeiBTMf99VHo9jB0uZW2b4BlVHPkuGPB5p33QDUsGPkEQ?e=Q0bFFV). Participants may upload labels using the following form until 11:59pm EST May 2nd, 2025: [Upload labels for review](https://docs.google.com/forms/d/e/1FAIpQLSeEYi_e15ePejBbt0VZ0mQW13Y3eDluPST4DFsPi8RmzvAIwQ/viewform). As submissions pass through the review process they will be made available [here](https://queensuca-my.sharepoint.com/:f:/g/personal/11rjh10_queensu_ca/EjDhNL_DbcVNhfTn10upNIIB8fiS1o4dL4xa6kwveryu3A?e=eaIyqR).   
  
#### Test Data:
Test data can be downloaded using the following link on May 5th, 2025 [here](https://queensuca-my.sharepoint.com/:u:/r/personal/11rjh10_queensu_ca/Documents/CREATE_Central_Line_Challenge/Test_Data.zip?download=1)

## Prepare Dataset for Training
Once all parts of the dataset have been downloaded for training, download code or clone this repository. Navigate to the location where the code is located and use the prepareDataset.py script to unpack and format your dataset. The script can be run by entering the following lines into your command prompt (replace all instances of UserName with your real username):  
```
conda activate createPytorchEnv  
cd <path_to_repository>  
python prepareDataset.py --compressed_location=C:/Users/UserName/Downloads --target_location=C:/Users/UserName/Documents/CreateChallenge --dataset_type=Train  
```  
To prepare the test set, follow the same steps, but change the --dataset_type flag to Test. The process is the same for the unlabelled data except --dataset_type should be "Unlabelled"  
  
If the code is executed correctly, you should see a new directory in your target location called either Training_Data or Test_Data. These directories will contain a set of subdirectories (one for each video) that contain the images. Within the main folder you will also see a csv file that contains a compiled list of all images and labels within the dataset. (Note: there will not be any labels for the test images).  

## Training the networks
Begin by activating your conda environment:
```
conda activate createPytorchEnv
```
Next select which network you would like to run. 
  
One baseline network has been provided for each subtask of the challenge:  
### Subtask 1: Surgical tool localization/ detection
Baseline network folder: Tool Detection    
Model: Yolo-v5   
Inputs: single image    
Outputs: list of dictionaries with the form {'class': classname, 'xmin': int, 'xmax': int, 'ymin': int, 'ymax': int, 'conf': float}  
  
Train the network (replace paths as necessary):
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Tool_Detection/Train_YOLOv5.py --save_location=C:/Users/SampleUser/Documents/toolDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Training_Data/Training_Data.csv
```
#### Required flags:
--save_location:   The folder where the trained model and all training information will be saved  
--data_csv_file:   File containing all files and labels to be used for training  
  
#### Optional flags:
--batch_size: Number of images included in each batch  
--epochs: Number of epochs  
--val_percentage: Percent of data to be used for validation  
--balance: Balance samples for training  
--patience: Epochs to wait for no observable improvement for early stopping of training  
--optimizer: Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.  
--lr0: Initial learning rate  
--lrf: Final learning rate  
--close_mosaic: Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.  
--freeze: Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.  
--box: Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.  
--cls: Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.  
--dfl: Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.  
--imgsz: Size of input images as integer  
--workers: Number of worker threads for data loading  
--device: device to use for training / testing (default 'cuda')  
--include_blank: Include images that have no labels for training  
--val_iou_threshold: IoU threshold used for NMS when evaluating model  
--val_confidence_threshold: Confidence threshold used for NMS when evaluating model  

### Subtask 2: Workflow recognition
Baseline network folder: Task Recognition    
Model: ResNet50 + Recurrent LSTM model  
Inputs: sequence of consecutive images  
Outputs: (10,1) softmax output  
  
Train the network (replace paths as necessary):
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Task_Recognition/Train_CNN_LSTM.py --save_location=C:/Users/SampleUser/Documents/taskDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Training_Data/Training_Data.csv
```
#### Required flags:
--save_location:   The folder where the trained model and all training information will be saved  
--data_csv_file:   File containing all files and labels to be used for training  
#### Optional flags:
--cnn_epochs: Number of epochs to run in training the cnn (int)  
--lstm_epochs: Number of epochs to run in training the lstm network(int)  
--validation_percentage: The percentage of data to be reserved for validation (float, range: 0-1)  
--cnn_batch: Number of images to be processed per batch (int)  
--lstm_batch: Number of images to be processed per batch (int)  
--cnn_lr: Learning rate used for loss function optimization for cnn (float)  
--lstm_lr: Learning rate used for loss function optimization for lstm network (float)  
--balance_cnn: Balance the number of samples from each class for training CNN (bool, True or False)  
--balance_lstm: Balance the number of samples from each class for training LSTM (bool, True or False)  
--augment_cnn: Use augmentations when training CNN (bool, True or False)  
--cnn_features: Number of features in last layer of CNN before the final softmax  
--lstm_sequence_length: number of consecutive images used as a single sequence (int)  

### Subtask 3: Ultrasound probe segmentation using foundation models
No baseline code is provided for subtask 3. However, you can start by exploring models such as SAM (Segment Anything Model). Your input should be a single image or a sequence of consecutive images. Your code should produce a single NumPy array for each segmentation (each NumPy array should be # of slices x image height x image width) 

## Generating test predictions
Each network folder contains a script for generating the predictions on test data. This script is run the same way as the training scripts. For example:
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Task_Recognition/generateTestPredictions.py --save_location=C:/Users/SampleUser/Documents/taskDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Test_Data/Test_Data.csv
```
Each of these scripts will generate an individual csv file with the prediction results. Should you choose to attempt more than 1 sub-task you must combine these files into one single csv.  
