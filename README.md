# Experiment 12 -- Image Processing and Object Detection

## Name

**Roshan V**

## Register Number

**212225240039**

------------------------------------------------------------------------

## Experiment Overview

This experiment contains three image-processing and computer-vision
tasks using **Python, OpenCV, NumPy, and Matplotlib**:

1.  ROI Segmentation in an Image using Bitwise AND
2.  Handwriting Detection in an Image
3.  Object Detection with Labels in an Image using MobileNet-SSD



------------------------------------------------------------------------

# 1. ROI Segmentation in an Image using Bitwise AND

## Aim

To segment and display a selected **Region of Interest (ROI)** from an
image using the OpenCV `bitwise_and()` operation.

## Requirements

-   Python 3
-   OpenCV
-   NumPy
-   Matplotlib
-   Input image: `gtr.jpg`

## Procedure

1.  Import OpenCV, NumPy, and Matplotlib.
2.  Read the input image using `cv2.imread()`.
3.  Convert the image from BGR to RGB for display.
4.  Display the original image.
5.  Define the required Region of Interest (ROI).
6.  Create a blank mask with the same dimensions as the original image.
7.  Place the selected ROI onto the mask.
8.  Perform bitwise AND between the original image and the mask.
9.  Convert the segmented image to RGB.
10. Display the segmented ROI.

## Important Code

``` python
image = cv2.imread('gtr.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

roi = image[100:420, 200:550]

mask = np.zeros_like(image)
mask[100:420, 200:550] = roi

segmented_roi = cv2.bitwise_and(image, mask)
```

## Output Screenshot

> **Paste the output screenshot below.**

**Output -- Original Image**

<img width="346" height="548" alt="image" src="https://github.com/user-attachments/assets/8d65839b-0b7c-4288-8746-f2cd23565424" />


**Output -- Segmented ROI**

<img width="288" height="516" alt="image" src="https://github.com/user-attachments/assets/33dbf58f-7522-4115-a81e-7822d80ce0fa" />


------------------------------------------------------------------------

# 2. Handwriting Detection in an Image

## Aim

To detect handwriting-like regions in an image using **grayscale
conversion, Gaussian blur, Canny edge detection, contour detection, and
bounding boxes**.

## Requirements

-   Python 3
-   OpenCV
-   NumPy
-   Matplotlib
-   Input image: `gtr.jpg`

## Procedure

1.  Read the input image.
2.  Convert the image from BGR to RGB for display.
3.  Display the original image.
4.  Convert the image to grayscale.
5.  Apply Gaussian blur to reduce noise.
6.  Apply the Canny edge detector.
7.  Find contours in the edge image.
8.  Filter contours based on area.
9.  Draw bounding boxes around the detected regions.
10. Display the handwriting detection result.

## Important Code

``` python
image = cv2.imread('gtr.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

edges = cv2.Canny(blurred_image, 50, 150)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result_image = image.copy()

for contour in contours:
    if cv2.contourArea(contour) > 50:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
```

## Output Screenshot

> **Paste the output screenshot below.**

**Output -- Canny Edge Detection**

<img width="287" height="515" alt="image" src="https://github.com/user-attachments/assets/7ec62001-c285-46ac-a7ba-2cde56b0f3b5" />


**Output -- Handwriting Detection**

<img width="286" height="508" alt="image" src="https://github.com/user-attachments/assets/f27bc539-7ce8-4302-9861-6789d7a77857" />


------------------------------------------------------------------------

# 3. Object Detection with Labels using MobileNet-SSD

## Aim

To detect objects in an image and display their corresponding class
labels using the **MobileNet-SSD pretrained deep learning model**.

## Requirements

-   Python 3
-   OpenCV
-   NumPy
-   Matplotlib
-   `gtr.jpg`
-   `deploy.prototxt`
-   `mobilenet_iter_73000.caffemodel`

## Model Files

The MobileNet-SSD model files are obtained from the following
repository:

https://github.com/chuanqi305/MobileNet-SSD

The required files are:

``` text
deploy.prototxt
mobilenet_iter_73000.caffemodel
```

Keep the model files in the same working folder as the notebook, or
provide their correct paths in the program.

## Procedure

1.  Import OpenCV, NumPy, and Matplotlib.
2.  Set the paths for the MobileNet-SSD configuration and weights files.
3.  Load the pretrained model.
4.  Define the object class labels.
5.  Read the input image.
6.  Obtain the image height and width.
7.  Convert the image to RGB for displaying.
8.  Create a blob using `cv2.dnn.blobFromImage()`.
9.  Set the blob as the model input.
10. Perform a forward pass using `net.forward()`.
11. Check the confidence value for each detection.
12. Accept detections with confidence greater than `0.5`.
13. Obtain the detected class label and bounding-box coordinates.
14. Draw rectangles and labels around detected objects.
15. Display the final object-detection result.

## Object Classes

The program uses the following MobileNet-SSD class labels:

    Index Class
  ------- -------------
        0 background
        1 aeroplane
        2 bicycle
        3 bird
        4 boat
        5 bottle
        6 bus
        7 car
        8 cat
        9 chair
       10 cow
       11 diningtable
       12 dog
       13 horse
       14 motorbike
       15 person
       16 pottedplant
       17 sheep
       18 sofa
       19 train
       20 tvmonitor

## Important Code

``` python
config_file = 'deploy.prototxt'
weights = 'mobilenet_iter_73000.caffemodel'

net = cv2.dnn.readNetFromCaffe(config_file, weights)

blob = cv2.dnn.blobFromImage(
    image,
    0.007843,
    (300, 300),
    127.5
)

net.setInput(blob)
detections = net.forward()
```

The detection threshold used in the notebook is:

``` python
if confidence > 0.5:
```

## Output Screenshot

> **Paste the final MobileNet-SSD output screenshot below.**

### Output -- Object Detection with Labels

<img width="415" height="519" alt="image" src="https://github.com/user-attachments/assets/d4652c7a-d7e3-4003-8da0-e54ba9b7e182" />


------------------------------------------------------------------------

# Result

Thus, the following image-processing and computer-vision operations were
successfully implemented:

-   **ROI segmentation** using Bitwise AND.
-   **Handwriting detection** using Canny edge detection and contours.
-   **Object detection with labels** using the MobileNet-SSD pretrained
    model.

The outputs are displayed using Matplotlib.

------------------------------------------------------------------------

# Folder Structure

Keep the required files organized as follows:

``` text
Experiment/
│
├── Ex 12(1).ipynb
├── catimg.jpg
├── plant.png
├── hand text.jpg
├── deploy.prototxt
└── mobilenet_iter_73000.caffemodel
```

------------------------------------------------------------------------

# Software Used

  Software / Library   Purpose
  -------------------- -------------------------------------------
  Python               Programming language
  OpenCV               Image processing and DNN object detection
  NumPy                Array and numerical operations
  Matplotlib           Image visualization
  MobileNet-SSD        Pretrained object detection model

------------------------------------------------------------------------

# Conclusion

The experiment demonstrates basic image processing and computer-vision
techniques using Python and OpenCV. ROI segmentation isolates a selected
region of an image, the handwriting detection section identifies edge
and contour-based regions, and MobileNet-SSD performs pretrained object
detection with class labels and bounding boxes.

------------------------------------------------------------------------

## Output Section -- Quick Reference

### Experiment 1 -- ROI Segmentation

**Screenshot:**

<img width="288" height="516" alt="image" src="https://github.com/user-attachments/assets/06a0efb9-5beb-4dc3-989e-7d6ce1f418e1" />


------------------------------------------------------------------------

### Experiment 2 -- Handwriting Detection

**Screenshot:**

<img width="286" height="508" alt="image" src="https://github.com/user-attachments/assets/16ae8217-2d5a-4a1d-bf67-c66d993e8ac0" />

------------------------------------------------------------------------

### Experiment 3 -- MobileNet-SSD Object Detection

**Screenshot:**

<img width="415" height="519" alt="image" src="https://github.com/user-attachments/assets/51559ba2-70fe-4d87-aad8-42c5d25d1f0a" />

------------------------------------------------------------------------

**End of Experiment 12**
