from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from models import *
import torchvision.transforms as transforms
from PIL import Image
import io
from fastapi.responses import HTMLResponse
from test.pre_train_test import*

# Khởi tạo FastAPI app
app = FastAPI()

import yaml

# Đọc file config.yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model = config["model"]
if model == "VGG":
    net = VGG('VGG19')
elif model == "ResNet":
    net = ResNet18()
elif model == "PreActResNet":
    net = PreActResNet18()
elif model == "GoogLeNet":
    net = GoogLeNet()
elif model == "DenseNet":
    net = DenseNet121()
elif model == "ResNeXt":
    net = ResNeXt29_2x64d()
elif model == "MobileNet":
    net = MobileNet()
elif model == "MobileNetV2":
    net = MobileNetV2()
elif model == "DPN":
    net = DPN92()
elif model == "ShuffleNetG2":
    net = ShuffleNetG2()
elif model == "SENet":
    net = SENet18()
elif model == "ShuffleNetV2":
    net = ShuffleNetV2(1)
elif model == "EfficientNet":
    net = EfficientNetB0()
elif model == "RegNet":
    net = RegNetX_200MF()
elif model == "SimpleDLA":
    net = SimpleDLA()
# Load mô hình từ checkpoint
checkpoint = torch.load('./checkpoint/ckpt.pth')  # Tải checkpoint
net.load_state_dict(checkpoint['net'])
net.eval()  # Đặt mô hình ở chế độ đánh giá (eval)

# Các lớp của CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

# Định nghĩa dữ liệu đầu vào
class InputData(BaseModel):
    data: List[float]  # Dữ liệu đầu vào là một danh sách các số (ví dụ: đặc trưng của mô hình)

# Các phép biến đổi ảnh cho CIFAR-10
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize ảnh về kích thước 32x32
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Chuẩn hóa ảnh
])

# Trang chủ để phục vụ giao diện HTML
@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
    <html>
    <head>
        <title>Chọn ảnh và Dự đoán CIFAR-10</title>
        <script>
            async function handleSubmit(event) {
                event.preventDefault();  // Ngừng hành động mặc định của form

                const formData = new FormData();
                const fileInput = document.querySelector("input[type='file']");
                formData.append("file", fileInput.files[0]);  // Lấy file ảnh từ input

                // Hiển thị ảnh trước khi gửi
                const file = fileInput.files[0];
                const reader = new FileReader();
                reader.onloadend = function() {
                    // Tạo một thẻ img để hiển thị ảnh
                    const imageElement = document.createElement("img");
                    imageElement.src = reader.result;
                    imageElement.style.maxWidth = "300px";  // Giới hạn kích thước ảnh
                    imageElement.style.maxHeight = "300px";
                    document.getElementById("uploadedImage").innerHTML = "";
                    document.getElementById("uploadedImage").appendChild(imageElement);
                };
                reader.readAsDataURL(file);  // Đọc file ảnh

                // Gửi formData đến endpoint /predict
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();  // Nhận kết quả trả về từ server

                // Kiểm tra nếu có lỗi từ server
                if (data.error) {
                    document.getElementById("result").innerHTML = "<h3>Lỗi: " + data.error + "</h3>";
                } else {
                    // Hiển thị kết quả dự đoán
                    document.getElementById("result").innerHTML = "<h3>Ảnh thuộc lớp: " + data.prediction + "</h3>";
                }
            }
        </script>
    </head>
    <body>
        <h2>Chọn một ảnh để dự đoán lớp (CIFAR-10)</h2>
        <form id="predictForm" onsubmit="handleSubmit(event)">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Dự đoán</button>
        </form>

        <div id="uploadedImage" style="margin-top: 20px;"></div>  <!-- Hiển thị ảnh đã tải lên -->
        <div id="result" style="margin-top: 20px;"></div>  <!-- Hiển thị kết quả dự đoán -->
    </body>
</html>

    """

# Endpoint để thực hiện dự đoán
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        # Áp dụng các phép biến đổi để chuẩn bị dữ liệu đầu vào cho mô hình
        test_image_size(data_preparation, image)
        test_image_mode(data_preparation, image)
        image = transform(image).unsqueeze(0)  # Thêm batch dimension
        # Thực hiện dự đoán
        with torch.no_grad():
            output = net(image)  # Dự đoán lớp của ảnh
            _, predicted_class = torch.max(output, 1)  # Lấy lớp dự đoán với xác suất cao nhất

        # Trả về kết quả dưới dạng JSON
        predicted_label = CIFAR10_CLASSES[predicted_class.item()]
        return {"prediction": predicted_label}
    
    except Exception as e:
        # Nếu có lỗi, trả về thông báo lỗi
        return {"error": str(e)}

