## Caffe model to tensorflow
#### Contents:
- `task.txt`: Lists the task instructions
- `Transfer-Learning.ipynb`: Jupyter notebook containing the experiment. Contains one of the experiments that I could submit due to time constraints. (uses subset of the whole dataset, dictated by the variable named `tiny`)
- `vgg.py`: Implementation of VGG16 model and corresponding functions
- `utils.py`: Helper functions for data parsing, loading etc.
- `fc7_[train/val]_feats[_large].npy`: Stored feature descriptors
- `caffe-tensorflow/`: The open-source tool for model conversion
- `vgg_face_caffe/`: The pretrained model
- `Data/`: The dataset containing the mages, labels and metadata
- `saved_model/`: Contains the saved model after hyper-parameter search

  *Structure*:
 - `Data/fold_frontal_[0/1/2/3/4]_data.txt`
 - `Data/aligned/.../*.jpg`

> ###### To-Do
> - Save features for all images
> - Save with mapping from features to image path
> - Simple baseline like SVM or Logistic Regression
> - Compare performance with fine-tuning

### Description:
The pretrained network is used as a fixed (non-trainable) feature extractor. These features are passed through another, relatively simpler neural network, which is trained as a classifier. All results are accessible in the jupyter notebook mentioned above (All the code has been written from scratch for the purpose of this project/experiment, except the code for plotting a confusion matrix, and the code to convert Caffe model to Tf format)

### Initial Setup:
##### Model
- Download pretrained model [weights](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
- Use [open-source tool](https://github.com/ethereon/caffe-tensorflow) for converting the saved weights to `.npy` format. (Note: a few edits were required, due to a `Caffe` version conflict). `vgg16.npy` is the output of the process.

##### Data
- Download the [dataset](http://www.openu.ac.il/home/hassner/Adience/data.html#agegender) from the linked ftp server, this contains images, gender labels and metadata as `.txt` files.
- Write wrapper code to locate and load the data for feature extraction. `utils.py` contains the corresponding code.
- The different `.txt` files had already been split for training vs. validation set.

> ###### To-Do
> - Normalize data to make it zero mean
> - Add data augmentation functionality
> - Add randomness in choosing train-val split

### Feature Extraction:
- `vgg.py` contains code to create the VGG16 model based on the architecture described in the [paper](https://arxiv.org/abs/1409.1556)
- All code was written from scratch, by referring to [TensorFlow documentation](https://www.tensorflow.org/api_docs/).
- Functionality has been added to be used to either train from scratch, or load pretrained weights from a specified file.
- The `extract_features` function returns the `fc6` and `fc7` layer features. (Note: I planned to run experiments using both layers' features and compare performance)

> ###### To-Do
> - Add functionality to run on part of dataset and save/load the features

### Transfer Learning:
- Build a simple one layer neural network classifier that would take the extracted features as input and classify gender
 - Extracted features go through a `ReLU` to a layer with `hidden_dim` neurons
 - `Batch_norm`, `ReLU` and `dropout` applied to the output
 - This output goes into the final layer with 2-dimensional softmax logits output, with a cross-entropy loss with a one-hot encoding of the labels
- Includes functionality to train the classifier, evaluate performance on validation set after a few epochs, track losses and accuracy, return actual prediction

> ###### To-Do
> - Add more evaluation metric like precision-recall, ROC curve, qualitative error analysis, saliency maps, functionality to visualize weights
> - Hyperparameter tuning for improved performance on the training and validation dataset
> - Saving best model functionality
> - Add logging or .txt logs

### Tasks
###### Final Tensorflow classifier model trained on the gender dataset (architecture and weights)
- *pending*

###### Code for training and evaluating the model
- Refer to `Transfer-Learning.ipynb`

###### Results/metrics(for all the classes and overall) obtained for the trained model
- Refer to `Transfer-Learning.ipynb`

###### Readme file briefly listing the steps taken and the steps to run your code (training and evaluation)
- This file
