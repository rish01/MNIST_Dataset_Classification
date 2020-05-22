# CPSC340 Final Exam Part 1 - MNIST Dataset Classification

The goal of this part of the final exam is to classify each of the 10,000 MNIST 28x28 pixels images of handwritten digits as accurately as possible using several different machine learning models implemented from scratch. The models tested include KNN, linear regression, SVM, MLP and CNN. The report for this task is present [here](https://github.com/rish01/CPSC340_MNIST_Dataset_Classification/blob/master/report/finalExam_Q1.pdf).

## Installation
1. Clone the repo on your system. 
2. Install Anaconda from [here](https://www.anaconda.com/products/individual) which will be used to setup virtual environment for the project.
3. Ensure conda is installed and in the PATH in your system's environment variables. 
4. Create a virtual environment using conda by typing the entering this command in your terminal/command prompt: <br />
```conda create -n mnist_classification_venv python=3.6 anaconda```
5. Activate the newly created virtual environment using this command:<br />
```source activate mnist_classification_venv```
6. Navigate to the folder containing the repo (if not already there).
7. Install the required Python packages to the virtual environment using this command:<br />
```conda install -n mnist_classification_venv -r requirements.txt```
8. Link the python.exe file present in the conda virtual environment folder to the interpreter of your IDE. 

<br />

## Testing a Model
Before running main.py file, please change the value of *model* variable present on line 36 to one of *KNN, LINEAR REGRESSION, SVM, MLP or CNN*.
