# Multi-Framework Pose Keypoint Endpoint with FastAPI

This repository provides a FastAPI endpoint that generates pose keypoints from various frameworks, including Detectron, MediaPipe, YOLO, and PoseNet.

## Key Features:

Unified API: Access multiple pose estimation frameworks through a single, consistent API.
Performance Comparison: Easily assess the performance of different frameworks for your specific use cases.
Flexibility: Integrate the endpoint seamlessly into your applications and workflows.
## Installation:

Clone this repository:
Bash
git clone https://github.com/antonysamuel/pose_estimation_framework.git
Use code with caution. Learn more
Install dependencies:
Bash
cd pose_estimation_frameowork
pip install -r requirements.txt
Use code with caution. Learn more
## Usage:

Run the FastAPI server:
Bash
uvicorn main:app --reload
Use code with caution. Learn more
Access the endpoint using your preferred HTTP client (e.g., Postman, curl):
Bash
curl -X POST http://127.0.0.1:8000/pose \
    -H "Content-Type: image/jpeg" \
    -d @image.jpg
Use code with caution. Learn more
## Available Frameworks:

Detectron
MediaPipe
YOLO
PoseNet
## Customization:

Configure the default framework in config.py.
Extend the endpoint with additional frameworks as needed.
## Contributing:

We welcome contributions! Please see the CONTRIBUTING.md file for more details.

## License:

This project is licensed under the MIT License. See the LICENSE file for more details.

