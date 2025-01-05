from preprocess_data import preprocess
import pytest
import torch
from PIL import Image
from torchvision import transforms
import config

@pytest.fixture
def data_preparation():
    return preprocess()

# kiểm tra kích thước hình ảnh
def test_image_size(data_preparation, image):
    def validate_image_size(image):
        if image.size[0] < 224 or image.size[1] < 224:
            raise ValueError("kích thước ảnh ít nhất là 224x224")


    # Kiểm tra kích thước ảnh trước khi xử lý
    validate_image_size(image)

    # Sau khi xử lý, ảnh phải có kích thước (224, 224)
    transformed_image = data_preparation(image)
    assert transformed_image.shape[1:] == INPUT_SIZE, "Ảnh sau khi resize không đạt kích thước 224x224."


# Test case kiểm tra định dạng ảnh
def test_image_mode(data_preparation, image):
    def validate_image_mode(image):
        if image.mode != 'RGB':
            raise ValueError("Ảnh phải có định dạng RGB.")

    # Đưa ảnh thực vào kiểm tra
    # image = Image.open("path_to_your_image.jpg")  # Đường dẫn tới ảnh cần kiểm tra

    # Kiểm tra định dạng ảnh trước khi xử lý
    validate_image_mode(image)

    # Sau khi xử lý, đảm bảo ảnh là Tensor với 3 kênh (RGB)
    transformed_image = data_preparation(image)
    assert transformed_image.shape[0] == 3, "Ảnh không có 3 kênh (RGB) sau khi chuyển đổi."