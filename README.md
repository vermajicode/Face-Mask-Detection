# Face-Mask-Detection

This project demonstrates a **Face Mask Detection System** using a deep learning model. The system identifies whether a person in an input image is wearing a mask or not. It leverages pre-trained models, OpenCV for image processing, and TensorFlow/Keras for predictions.

## Features
- **Input Image Handling**: Allows users to input an image and visualize it.
- **Image Preprocessing**: Resizes and normalizes the input image for model compatibility.
- **Prediction**: Classifies the image into two categories:
  - Wearing a Mask
  - Not Wearing a Mask
- **Interactive Interface**: Uses Google Colab's interactive capabilities for predictions.

## Requirements
To run the notebook, ensure you have the following installed:
- Python 3.7 or higher
- TensorFlow/Keras
- OpenCV
- NumPy
- Google Colab (for interactive use)

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/vermajicode/face-mask-detection.git
   cd face-mask-detection
Open the notebook Face_Mask_Detection.ipynb in Google Colab or Jupyter Notebook.
Ensure all required libraries are installed.
Load your trained model into the notebook.
Run the code cells in sequence to:
Load and preprocess the image.
Get predictions from the model.
Display the results.
Function for Reuse
The notebook includes a reusable function for predicting mask status:

python
Copy code
def predict_mask(model):
    """
    Predicts if the person in the image is wearing a mask or not using the provided model.

    Parameters:
    model: The pre-trained model for mask detection.

    Returns:
    None
    """
    input_image_path = input('Path of the image to be predicted: ')
    input_image = cv2.imread(input_image_path)
    
    if input_image is None:
        print("Error: Could not read the image. Please check the path.")
        return

    cv2_imshow(input_image)
    input_image_resized = cv2.resize(input_image, (128, 128))
    input_image_scaled = input_image_resized / 255.0
    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])
    input_prediction = model.predict(input_image_reshaped)
    print("Prediction Probabilities:", input_prediction)
    input_pred_label = np.argmax(input_prediction)
    print("Predicted Label:", input_pred_label)
    
    if input_pred_label == 0:
        print('The person in the image is wearing a mask.')
    else:
        print('The person in the image is not wearing a mask.')
Example Output
lua
Copy code
Path of the image to be predicted: /path/to/image.jpg
[Image Displayed]
Prediction Probabilities: [[0.13046394 0.98431385]]
Predicted Label: 1
The person in the image is not wearing a mask.
Dataset
The system is trained on two types of dataset face mask and without face mask images. Ensure the dataset is preprocessed and split into training and testing sets before training the model.

Acknowledgments
TensorFlow/Keras: For deep learning framework.
OpenCV: For image handling and processing.
Google Colab: For interactive model deployment.
