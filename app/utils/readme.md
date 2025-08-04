# Image Colorization Project

## Overview

This project is an image colorization solution built with deep learning. It turns grayscale images into colorized images using convolutional neural networks (CNN) and architectures like **U-Net** and **EfficientNetB0**. Developed in Python with **TensorFlow**, **FastAPI**, and Docker, it provides a fully-functional API for image colorization.

---

## Features

- **Model Architectures**: Support for U-Net and EfficientNetB0 for colorization tasks.  
- **Flexible Input**: Accepts both grayscale images and Lab-space images.  
- **Pretrained Models**: Includes pretrained weights for faster inference.  
- **Web API**: Serve via a **FastAPI** RESTful API for real-time predictions.  
- **Dockerized**: Easily deploy with Docker for scalability.  
- **Model Conversion**: Export to ONNX, TensorFlow SavedModel, TFLite, and H5 formats.

---

## Table of Contents

- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [API Endpoints](#api-endpoints)  
- [Model Training](#model-training)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Installation

### Prerequisites

- **Python 3.8+**  
- **Docker** (optional, for deployment)  
- **TensorFlow**  
- **FastAPI**  

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/WaleedAnas/image-colorization.git
   cd image-colorization


## Usage

### Running the API

Start the FastAPI server locally:

```bash
uvicorn app.main:app --reload
```

Server runs at `http://127.0.0.1:8000`.

### Making Predictions

Send a POST request to `/predict` with your image file.

**Using `curl`:**

```bash
curl -X POST -F "file=@path/to/image.jpg" \
  http://127.0.0.1:8000/predict \
  -o colorized_image.jpg
```

**Using Python (`requests`):**

```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("path/to/image.jpg","rb")}
response = requests.post(url, files=files)

with open("colorized_image.jpg","wb") as f:
    f.write(response.content)
```

---

## Project Structure

The project is organized as follows:

### Breakdown of the Project Structure

- **`app/`**: The main FastAPI application.
  - **`main.py`**: The entry point for the FastAPI application.
  - **`models/`**: This folder contains the deep learning models like U-Net, EfficientNetB0, etc.
  - **`utils/`**: Helper functions for preprocessing and postprocessing images.
  - **`routes/`**: Definitions of API endpoints.
  - **`dependencies.py`**: Contains FastAPI dependency injections such as request logger.
  - **`config.py`**: Configuration file for model paths and other settings.

- **`data/`**: Folder to store datasets.
  - **`raw/`**: Contains the unprocessed raw dataset.
  - **`processed/`**: Contains preprocessed data ready for model training.

- **`scripts/`**: Utility scripts for various tasks.
  - **`train.py`**: Main script to train the models.
  - **`convert_to_onnx.py`**, **`convert_to_tflite.py`**: Scripts to convert the trained models to different formats (ONNX, TFLite).

- **`Dockerfile`**: Instructions to build the Docker image for deployment.

- **`requirements.txt`**: List of required Python packages for the project.

- **`README.md`**: The file you're reading now, containing the project documentation.

---

This version is clearer, with indentation that follows the directory structure and additional descriptions to explain each part of the project. This will make it more engaging and easier to understand for anyone navigating the repository.

## API Endpoints

### `POST /predict`

- **Description**: Colorize a grayscale image.
- **Parameters**:

- `file` (form-data): JPEG or PNG image.
- **Response**: Colorized image in JPEG format.

**Example:**

```bash
curl -X POST -F "file=@path/to/image.jpg" \
  http://127.0.0.1:8000/predict \
  -o colorized_image.jpg
```

---

## Model Training

Train your model using the script in `scripts/`:

```bash
python scripts/train.py \
  --model unet \
  --data_path path/to/dataset \
  --epochs 50
```

- `--model`: `unet` or `efficientnetb0`
- `--data_path`: Path to your dataset
- `--epochs`: Number of training epochs

**Note**: Organize your dataset with grayscale images in one folder and their color versions in another.

---

## Contributing

Contributions are welcome!

1. Fork the repo.
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request.

Please open an issue for bug reports or feature requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.\`\`\`
