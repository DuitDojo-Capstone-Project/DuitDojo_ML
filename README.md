# DuitDojo_ML
There are two models that the ML team has worked on, the first is image to text and the second is text classification
## Image-to-text Model:
1. Transfer learning from donut-model, which is a new OCR-free approach.
2. Using the CORD dataset to train model on indonesian receipt
3. Using the pytorch framework
4. the model takes an image input and output the item's name, quantity, price, and total price

## Demo:
Input image:   
(JPG)   
![image](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/blob/main/image%20to%20text/6.JPG)
Output:   
(JSON)   
![image](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/blob/main/image%20to%20text/Example%20Output.png)

### How to use the repo:
1. Fork the repo
2. To use the Image-to-text Model:
   - Change the hugginface repo (to store the model)
   - Run the notebook to train the model on CORD dataset

## text classification: 
In the text classification folder there are 3 folders consisting of dataset, notebook, model and API
- in the dataset folder there is a dataset that is used. This dataset was collected by us ourselves. This dataset consists of five labels, namely Food, Clothing, Utilites, Electronic and Others.
- in the notebook folder there is a notebook for model training
- in the model folder there is the model.h5, tokenizer and label encoder
- in the API folder there is API for text classification

### How to use the notebook:
1. open the notebook
2. Run the notebook

### How to use the API:
The steps are in the [README.md file](text%20classification/API) in the API folder

### Model performance
![image](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/assets/126539714/18927cda-b8ce-499b-941e-de2320663132)
![image](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/assets/126539714/55c0dd22-1b8f-44a7-a66f-6886b164f892)

on test data:
![image](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/assets/126539714/bd4953fd-1c75-4b87-b263-53e2d736d753)

