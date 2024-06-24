## **Dual-Domain Framework**
[Paper](https://doi.org/10.1038/s44172-023-00121-z)

[Github](https://github.com/ZhangJD-ong/Medical-image-reconstruction-and-synthesis)



## **Euivarience**
[Robust Equivarience Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.pdf)

[Github](https://github.com/edongdongchen/REI)


## **File descriptions**

### data_preprocessing

- pre_test_data: This code applies all the required preprocessing steps to the HECKTOR raw ct and pet training data:
  -   Normalisation [ct and pet]
  -   Reshaping (--> 128) [ct and pet]
  -   Image selection [ct and pet]
  -   Background/Artifact removal [ct only]



- pre_train_data: This code applies all the required preprocessing steps to the HECKTOR raw ct and pet testing data:

	- Normalisation [ct and pet]
	- Reshaping (--> 128) [ct and pet]
	- Background/Artifact removal [ct only]
	- Image selection [ct and pet]

- testing_file_write: This is a simple script to write a text file that makes a text file in the format for calling on processed test data

- training_file_write: This is a simple script to write a text file that makes a text file in the format for calling on processed training data

- file_delete: After noting the data to be removed in a note file for each modality, this script will go through the data and remove the desired files from the datasets.
