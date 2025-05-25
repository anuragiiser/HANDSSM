This folder contains :

=======================================
1. Folder Name : statute_detection
=======================================
* It is directly download from AILA 2019 track. 
* The elaborated details has been indicated in README.text file inside the folder.

=======================================
2. Folder Name : final_data
=======================================
* It is collection of . 
* The elaborated details has been indicated in README.text file inside the folder.

=======================================
3. Jupyter file : AILA_dataset.ipynb
=======================================

* It is the file to create the suitable dataset to run the main code for DATASET1.
* It create two csv file named : 
	train1.csv -> Train data for DATASET1
	test1.csv  -> Test data for DATASET1

=======================================
4. Jupyter file : final_dataset.ipynb
=======================================

* It is the file to create the suitable dataset to run the main code for DATASET2.
* It create two csv file named : 
	train2.csv -> Train data for DATASET2
	test2.csv  -> Test data for DATASET2

=======================================
5. text file : requirement.txt
=======================================

*It is a text file containinf all python packages needed to run our main file.
*We can install all that files just by running the script in this location:
	pip install -r requirements.txt

=======================================
6. text file : glove.6B.50d.txt
=======================================	

*It refers to a pre-trained word embedding model.
*That was trained on a large corpus of text using the GloVe (Global Vectors) algorithm. 
*The model contains 400,000 unique words, represented as vectors in a 50-dimensional space.
*Download from - https://www.kaggle.com/datasets/watts2/glove6b50dtxt

=======================================
7. python file : dataset.py
=======================================

*It is a stub code for my main python file to convert csv file to a dataset format suitable for HAN.


=======================================
8. python file : HANSSM.py
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




2023-05-17 20:33:20,859 model formed
2023-05-17 20:53:19,210 Epoch: 1/1, Lr: 0.1, Loss: 0.6408066153526306, Accuracy:  0.8747203579418344
2023-05-17 20:55:50,338 Prediction:
Precision: 0.4373601789709172 Recall : 0.5 F1_Score : 0.46658711217183774





