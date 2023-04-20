# Central_Line_Challenge
For those using the School of Computing GPU Server, you may skip directly to training the networks step. The required anaconda environment has already been created for you, and the dataset has been pre-downloaded and prepared for you in advance. The following instructions explain how to reproduce the setup on a local computer.

## Create conda environment    

For those wishing to run on a local computer:  
1. Ensure Anaconda has been installed on your device: https://www.anaconda.com/products/distribution  
> - during installation, make sure to select the option to add anaconda to your search path  
2. Create a new conda environment
```
conda create -n createKerasEnv python=3.9
```  
3. Follow the step-by-step instructions provided [here](https://www.tensorflow.org/install/pip) for installing tensorflow and configuring your GPU if necessary. When you reach the step where you pip install tensorflow, install tensorflow==2.9 instead of 2.12 as listed in the instructions. For those using the School of Computing resources, follow the instructions for a linux installation.  
4. Install the remaining requirements  
```
conda activate createKerasEnv  
pip install pandas, scikit-learn, matplotlib, opencv-python
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
Download links are password protected and will only be available until May 5th, 2023. Registered participants will receive the password via email on April 24th, 2022.  
  
#### Training Data:
Training data can be downloaded in 8 parts using the following links: [Part 1](https://tinyurl.com/25rwnvdc), [Part 2](https://tinyurl.com/59fa3dpu), [Part 3](https://tinyurl.com/mrdsv9za), [Part 4](https://tinyurl.com/53xh9t9b), [Part 5](https://tinyurl.com/3jydcujz), [Part 6](https://tinyurl.com/ms5sdsk8), [Part 7](https://tinyurl.com/36zxmukf), [Part 8](https://tinyurl.com/369hnxef)  

#### Unlabelled Data:
Unlabelled data can be found [here](https://tinyurl.com/4zwd2v9m). Participants may upload labels using the following form until 12:00pm EST (noon) May 4th, 2023: [Upload labels for review](https://forms.gle/jMYfoKLBD9VsyF1b9). As submissions pass through the review process they will be made available [here](https://tinyurl.com/4zwd2v9m).   
  
#### Test Data:
Test data can be downloaded using the following link on May 4th, 2023: [Test Data]()

## Prepare Dataset for Training
Once all parts of the dataset have been downloaded for training, download code or clone this repository. Navigate to the location where the code is located and use the prepareDataset.py script to unpack and format your dataset. The script can be run by entering the following lines into your command prompt (replace all instances of UserName with your real username):  
```
conda activate createKerasEnv  
cd <path_to_repository>  
python prepareDataset.py --compressed_location=C:/Users/UserName/Downloads --target_location=C:/Users/UserName/Documents/CreateChallenge --dataset_type=Train  
```  
To prepare the test set, follow the same steps, but change the --dataset_type flag to Test. The process is the same for the unlabelled data except --dataset_type should be "Unlabelled"  
  
If the code is executed correctly, you should see a new directory in your target location called either Training_Data or Test_Data. These directories will contain a set of subdirectories (one for each video) that contain the images. Within the main folder you will also see a csv file that contains a compiled list of all images and labels within the dataset. (Note: there will not be any labels for the test images).  

## Training the networks
Begin by activating your conda environment:
```
conda activate createKerasEnv
```
Next select which network you would like to run. 
  
One baseline network has been provided for each subtask of the challenge:  
### Subtask 1: Surgical tool localization/ detection
Baseline network folder: Tool Detection    
Model: Yolo-v3   
Inputs: single image  
Outputs: list of dictionaries with the form {'class': classname, 'xmin': int, 'xmax': int, 'ymin': int, 'ymax': int, 'conf': float}  

1. Download the backend weights: [Backend weights](https://tinyurl.com/y4s6zsa2)

2. Place weights in the Tool_Detection directory
  
3. Train the network (replace paths as necessary):
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Tool_Detection/Train_Yolov3.py --save_location=C:/Users/SampleUser/Documents/toolDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Training_Data/Training_Data.csv
```
#### Required flags:
--save_location:   The folder where the trained model and all training information will be saved  
--data_csv_file:   File containing all files and labels to be used for training  
  
Additional hyperparameters such as batch size, learning rate and number of epochs can be changed by modifying config.json

Custom anchor box ratios can be generated using gen_anchors.py:
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Tool_Detection/gen_anchors.py --saved_run_location=<Path to a previous training run>
```

### Subtask 2: Workflow recognition
Baseline network folder: Task Recognition    
Model: ResNet50 + Recurrent LSTM model  
Inputs: sequence of consecutive images  
Outputs: (10,1) softmax output  
  
Train the network (replace paths as necessary):
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Task_Recognition/Train_CNN.py --save_location=C:/Users/SampleUser/Documents/taskDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Training_Data/Training_Data.csv
```
#### Required flags:
--save_location:   The folder where the trained model and all training information will be saved  
--data_csv_file:   File containing all files and labels to be used for training  
#### Optional flags:
--num_epochs_cnn: Number of epochs to run in training the cnn (int)  
--num_epochs_lstm: Number of epochs to run in training the lstm network(int)  
--validation_percentage: The percentage of data to be reserved for validation (float, range: 0-1)  
--batch_size: Number of images to be processed per batch (int)  
--cnn_learning_rate: Learning rate used for loss function optimization for cnn (float)  
--lstm_learning_rate: Learning rate used for loss function optimization for lstm network (float)  
--balance_CNN_Data: Balance the number of samples from each class for training CNN (bool, True or False)  
--balance_LSTM_Data: Balance the number of samples from each class for training LSTM (bool, True or False)  
--loss_function: loss function to optimize during training (str)  
--metrics: metrics used to evaluate the model (str) 
--sequence_length: number of consecutive images used as a single sequence (int)  
--downsampling_rate: number of frames to skip when generating training image sequences, used as a form of data augmentation (int)  

## Generating test predictions
Each network folder contains a script for generating the predictions on test data. This script is run the same way as the training scripts. For example:
```
python C:/Users/SampleUser/Documents/Central_Line_Challenge/Task_Recognition/generateTestPredictions.py --save_location=C:/Users/SampleUser/Documents/taskDetectionRun1 --data_csv_file=C:/Users/SampleUser/Documents/Test_Data/Test_Data.csv
```
Each of these scripts will generate an individual csv file with the prediction results. Should you choose to attempt more than 1 sub-task you must combine these files into one single csv.  
