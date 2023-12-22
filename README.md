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

## Text Classification:

The Text Classification module contains:

- **Dataset Folder**: Includes the collected dataset with five labels: Food, Clothing, Utilities, Electronics, and Others.
- **Notebook Folder**: Contains the notebook for model training. here we create a multiclass classification model using Tensorflow.
- **Model Folder**: Consists of model.h5, tokenizer, and label encoder.
- **API Folder**: Contains the API using Flask for text classification.

### How to use the notebook:

1. Open the notebook.
2. Run the notebook.

### How to use the API:

Detailed steps can be found in the [API README](text%20classification/API).

### Model Performance:
There are several models in the notebook file. The model used is the first model with performance as follows

![Training Loss and Accuracy](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/assets/126539714/18927cda-b8ce-499b-941e-de2320663132)
![Validation Loss and Accuracy](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/assets/126539714/55c0dd22-1b8f-44a7-a66f-6886b164f892)

On Test Data:
![Test Data Performance](https://github.com/DuitDojo-Capstone-Project/DuitDojo_ML/assets/126539714/bd4953fd-1c75-4b87-b263-53e2d736d753)
