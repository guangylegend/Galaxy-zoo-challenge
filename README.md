Guang Yang (gy552) & Chen Zhang (cz1389)

## Description
We implemented several models. For your convinence, we placed each banch in a seperate directory. <br />
The directories are name as "GalaxyZoo-<method>"<br />
In each directory, <br />
-- download.sh     # A shell script allowing you to download the data used for the task. <br />
                   # Your need to export your cookie for the Kaggle challenge and put it in cookie.txt.<br />
                   # Or you can just download the data from the browser.<br />
-- main.py         # The skeleton of the training and validation.<br />
-- data.py         # Data preparation including preprocessing and augmentation.<br />
-- model.py        # The model definition.<br />
-- evaluate.py     # Generate the test output.<br />
  
The files are brought from assignment 3, therefore have similar command line argument.<br />

## Usage
To train the model, invoke
> python main.py --epochs 200 --model_name <model_name>

It will find the local minimums of the validation loss, and dump the model as <model-name>_<epoch>.pth<br />

To generate the test output, invoke
> python evaluate.py --model <model-name>_<epoch>.pth
