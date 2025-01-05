from mlops.src.preprocess_data import preprocess
import pytest
import torch
from PIL import Image
from torchvision import transforms
import mlops.src.config as config

@pytest.fixture
def data_preparation():
    return preprocess()

# kiểm tra kích thước hình ảnh
from PIL import Image
import torch
from torchvision import transforms

def test_image_size(data_preparation):
    def validate_image_size(image):
        # Kiểm tra kích thước ảnh, đảm bảo ít nhất 224x224
        if image.size[0] < 224 or image.size[1] < 224:
            raise ValueError("Kích thước ảnh phải ít nhất là 224x224.")

    image = Image.open(r"C:\Users\Nam Dao\Desktop\AI_SOICT_HUST\4_MLOPs\mlops\predata.png")

    # Chuyển đổi sang RGB nếu ảnh có nhiều hơn 3 kênh (RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Kiểm tra kích thước ảnh trước khi xử lý
    validate_image_size(image)

    # Sau khi xử lý, đảm bảo ảnh có kích thước (224, 224)
    transformed_image = data_preparation(image)
    assert transformed_image.shape[1] == 224 and transformed_image.shape[2] == 224, "Kích thước sau khi chuyển đổi không đúng."

def test_image_mode(data_preparation):
    def validate_image_mode(image):
        if image.mode != 'RGB':
            raise ValueError("Ảnh phải có định dạng RGB.")

    image = Image.open(r"C:\Users\Nam Dao\Desktop\AI_SOICT_HUST\4_MLOPs\mlops\predata.png")

    # Chuyển đổi sang RGB nếu ảnh có nhiều hơn 3 kênh (RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Kiểm tra định dạng ảnh trước khi xử lý
    validate_image_mode(image)

    # Sau khi xử lý, đảm bảo ảnh là tensor với 3 kênh (RGB)
    transformed_image = data_preparation(image)
    assert transformed_image.shape[0] == 3, "Ảnh không có 3 kênh (RGB) sau khi chuyển đổi."

