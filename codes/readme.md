=======================================
1. Jupyter file : AILA_dataset.ipynb
=======================================

* It is the file to create the suitable dataset to run the main code for DATASET1.
* It create two csv file named : 
	train1.csv -> Train data for DATASET1
	test1.csv  -> Test data for DATASET1

=======================================
2. Jupyter file : final_dataset.ipynb
=======================================

* It is the file to create the suitable dataset to run the main code for DATASET2.
* It create two csv file named : 
	train2.csv -> Train data for DATASET2
	test2.csv  -> Test data for DATASET2

=======================================
3. text file : requirement.txt
=======================================

*It is a text file containinf all python packages needed to run our main file.
*We can install all that files just by running the script in this location:
	pip install -r requirements.txt

=======================================
4 python file : dataset.py
=======================================

*It is a stub code for my main python file to convert csv file to a dataset format suitable for HAN.


=======================================
5. python file : HANSSM.py
=======================================

*This is the main file to create the model.
*When you run the code by instruction:
	python3 HANSSM.py

*Firstly it will give a choice to you for the dataset you want to use. Like 


*Choose the dataset you want to use:
         1 for DATASET1
         2 for DATASET2

*You choose according to your requirement. 
*At the end of the run it create a log file named 'HAN.log' for the result.
*It also create a model named 'HAN_model.pth' for further use.
