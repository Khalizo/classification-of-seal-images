# Classification of Seal Images

Full report can be found [here](Practical-2-Classification-of-Seal-Images-Report.final.pdf)

## Introduction

Monitoring seals and their different stages of development can be very useful for institutions that are concerned about species preservation and climate change. However, manually sifting through months of data can be time-consuming and labour intensive. Moreover, classifying aquatic animals poses several challenges such as background noise, distortion of images, the presence of other water bodies in images and occlusion. Lastly, real-world datasets often have issues with data imbalance [2], which creates a challenge for machine learning models as most algorithms are designed to maximise accuracy and reduce the error.

Interestingly, with the developments in machine learning and image feature extraction, seal classification can be automated. Additionally, sampling strategies, such as over- and undersampling, are extremely popular in tackling the problem of clas.s imbalance where either the minority class is oversampled, the majority class is undersampled or a combination of the two . Consequently, the practice of combining multiple models to form an ensemble has been shown to address the weaknesses of individual models, thus creating a greater single model overall. This study trains several machine learning models with different resampling techniques, to classify seals from imbalanced datasets of features extracted from images.


## Dataset
To gain some visual intuition of the Histogram of Orientated Gradient (HOG) features, I sliced the
first 900 columns and picked an image from each class found in the Y_train datasets (see Figure
1). Despite the images being in very low resolution, there are still some visual differences between
each class. For instance, the whitecoat image has less dark pixels when compared to the moulted
pup. Observing the visual differences between the classes can also be useful for trying to get an
intuitive understanding of why a classifier misclassifies a particular class.

## Visualising the dataset

### Class Distribution

### PCA and t-SNE visualisations

## Results


## Conclusions
