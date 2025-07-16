** YOLO11 Object Detection Model**

This project demonstrates the use of the Ultralytics YOLO11 model for object detection.

**Setup and Installation**

*   **Install Ultralytics:**
    ```bash
    !pip install ultralytics
    ```

**Usage**

1.  **Import YOLO:**
    ```python
    from ultralytics import YOLO
    ```

2.  **Load a Pretrained Model:**
    ```python
    model = YOLO("yolo11n.pt")
    ```
    *Note: "yolo11n.pt" refers to a nano-sized pretrained YOLO11 model.*

3.  **Perform Object Detection:**
    ```python
    results = model("/content/child.jpg")  # Predict on an image
    ```
    *Replace `"/content/child.jpg"` with the path to your desired image.*

4.  **Display Results:**
    ```python
    results[0].show()  # Display the detected objects and bounding boxes
    ```

**Example Output**

When running the detection on `/content/child.jpg`, the output indicates:

*   **Detected Objects:** 4 persons, 1 sports ball
*   **Processing Speed:**
    *   Preprocess: 7.4ms
    *   Inference: 175.0ms
    *   Postprocess: 2.5ms
    *   Total per image at shape (1, 3, 448, 640)

**Dependencies**

The `ultralytics` package automatically handles most dependencies. Key dependencies include:

*   `numpy`
*   `matplotlib`
*   `opencv-python`
*   `pillow`
*   `pyyaml`
*   `requests`
*   `scipy`
*   `torch`
*   `torchvision`
*   `tqdm`
*   `psutil`
*   `py-cpuinfo`
*   `pandas`
*   `ultralytics-thop`
*   Various `nvidia-cuda-*` libraries (for GPU acceleration with PyTorch)
