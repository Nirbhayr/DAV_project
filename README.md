# APTOS Blindness Dataset. Diabetic Retinopathy Detection using Fundus Images
This work was also submitted for DS 294 (CDS course at IISc) final project for year 2020-21

The goal of this work was to predict the correct stage of diabetic retinopathy (DR) based on eye fundus images. The problem is of `multi-class` prediction. The problem is also ordinal in nature i.e., stage 3 of DR is worse than stage 1 and hence this information must be captured by the model. 


The project is based on deep learning using the dataset from kaggle from the APTOS blindness challenege. The dataset can be viewed here, https://www.kaggle.com/c/aptos2019-blindness-detection/overview

The project was based on `transfer learning` approach and pre-trained ResNet50 (trained on ImageNet) was used for this task.
`Data Augmentation` such as random rotation, reflection, and radnom zoom were also applied to increase the robustness of the trained model. 

The network architecture used is shown below:
![image](https://user-images.githubusercontent.com/84196853/204462021-92596805-e9a3-4f52-9d2a-1ec62646972e.png)

This is implemented from the following paper which particpated in the actual competition and were ranked in top 50: https://arxiv.org/pdf/2003.02261v1.pdf

Briefly, three decoders are used on the features obtained from ResNet50. 
- Classification head
- Regression
- Ordinal Regression

Classification outputs a one-hot encoded vector. Regression outputs a real number between $[0, 4.5)$, which is then rounded off to nearest integer. The ordinal regression head aims to predict all categories up to the target. If a image falls into category $k$ then it automatically falls into categories from $0$ to $k-1$. Finally, the output of the network is obtained by using linear regression on the combined output of all the three heads. The output from oridnal regression headwas summed and classification head was passed through $argmax()$ before feeding to the dense layer for regression.  

The outputs of the training and validation are summarized in `Final Output` folder.  

## Thank you


