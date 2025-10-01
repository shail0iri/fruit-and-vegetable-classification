Fruit and Vegetable Classification System
A deep learning-based image classification system that can identify 36 different types of fruits and vegetables with high accuracy using TensorFlow and transfer learning.

📋 Project Overview
This project implements a convolutional neural network (CNN) using MobileNetV2 as the base model for classifying images of fruits and vegetables. The model achieves high accuracy in distinguishing between 36 different classes of produce items.

🚀 Features
Transfer Learning: Utilizes pre-trained MobileNetV2 model for feature extraction

Data Augmentation: Implements real-time data augmentation to improve model generalization

Grad-CAM Visualization: Includes explainable AI features to understand model predictions

Comprehensive Evaluation: Provides detailed performance metrics and confusion analysis

Easy Deployment: Includes prediction script for classifying new images

🛠️ Technical Details
Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet)

Custom Layers:

Global Average Pooling

Dense (128 units, ReLU)

Dense (128 units, ReLU)

Output (36 units, Softmax)

Input Size: 224×224×3

Optimizer: Adam

Loss Function: Categorical Crossentropy

Data Augmentation
Rotation: ±30 degrees

Zoom: ±15%

Width/Height Shift: ±20%

Shear: ±15%

Horizontal Flip: Enabled

📊 Dataset
The dataset contains images of 36 different fruits and vegetables organized into:

Training set: Primary training data

Validation set: Model tuning and early stopping

Test set: Final evaluation
📈 Performance
Test Accuracy: [Add your accuracy here]%

Training Time: ~X minutes/epoch on [Your Hardware]

Model Size: [Size of saved model]

🎯 Results
Training Progress
https://training_history.png

Confusion Matrix
https://confusion_matrix.png

2. Making Predictions
Use the provided prediction script:

python
from predict import predict_image

# Classify a new image
class_name, confidence = predict_image('path/to/your/image.jpg')
print(f"Predicted: {class_name} with {confidence:.2%} confidence")

🔧 Model Training
Key training parameters:

Batch Size: 32

Epochs: 5 (with early stopping)

Early Stopping: Patience of 2 epochs

Learning Rate: Default Adam (0.001)

📊 Evaluation Metrics
The project provides comprehensive evaluation including:

Accuracy scores

Classification report

Confusion matrix

Per-class accuracy

Misclassification analysis

🎨 Visualization Features
Sample Images: Display representative images from each class

Training Curves: Accuracy and loss over epochs

Confusion Matrix: Normalized and count-based versions

Grad-CAM: Heatmaps showing model attention

Misclassification Analysis: Examples of wrong predictions

🤖 Model Interpretation
The Grad-CAM implementation helps understand:

Which image regions influence predictions

Model decision-making process

Potential biases or focus areas

📝 Key Findings
The model achieves high accuracy on most classes

Some confusion occurs between visually similar items

Data augmentation significantly improves generalization

Transfer learning provides strong baseline performance

🚀 Future Improvements
Experiment with other base models (EfficientNet, ResNet)

Implement more sophisticated data augmentation

Add model ensemble techniques

Deploy as web application

Add support for real-time classification

