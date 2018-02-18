We implemented several models. For your convinence, we placed each banch in a seperate directory. 
The directories are name as "GalaxyZoo-<method>"
In each directory, 
  | 
  |-- download.sh     # A shell script allowing you to download the data used for the task. 
  |                   # Your need to export your cookie for the Kaggle challenge and put it in cookie.txt.
  |                   # Or you can just download the data from the browser.
  |
  |-- main.py         # The skeleton of the training and validation.
  |-- data.py         # Data preparation including preprocessing and augmentation.
  |-- model.py        # The model definition.
  |-- evaluate.py   # Generate the test output.
  
The files are brought from assignment 3, therefore have similar command line argument.

To train the model, invoke
> python main.py --epochs 200 --model_name <model_name>
It will find the local minimums of the validation loss, and dump the model as <model-name>_<epoch>.pth

To generate the test output, invoke
> python evaluate.py --model <model-name>_<epoch>.pth
