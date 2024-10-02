import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# 모델 클래스 정의 (기존 CRNN 모델)
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, None))  # 가변적인 가로 크기를 유지한 채 세로 크기를 1로 고정
        )
        
        self.rnn = nn.LSTM(256, nh, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(nh * 2, nclass)
    
    def forward(self, x):
        conv = self.cnn(x)  # CNN 통과 후 출력 크기 확인
        b, c, h, w = conv.size()  # batch_size, channels, height, width
        
        # RNN에 입력을 맞추기 위해 크기 조정
        conv = conv.squeeze(2)  # (batch_size, channels, width)로 크기 변환 (세로 크기를 제거)
        conv = conv.permute(0, 2, 1)  # (batch_size, width, channels)로 순서 변경
        
        output, _ = self.rnn(conv)  # RNN 통과
        
        # 시퀀스의 마지막 출력만 사용하여 손실 계산
        output = output[:, -1, :]  # (batch_size, num_classes) 크기로 만듦
        output = self.fc(output)  # 최종 출력
        
        return output

# 모델 로드 함수
def load_model(model_path, n_classes):
    model = CRNN(imgH=128, nc=3, nclass=11, nh=256)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 모델을 평가 모드로 설정
    return model

# 이미지 예측 함수
def predict_image(model, image_path, transform, device):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 모델을 통해 예측
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)
    
    return predicted.item()

# 네모 박스를 그리는 함수
def draw_bounding_box(image_path, label):
    # OpenCV로 이미지 로드
    image = cv2.imread(image_path)
    
    # 네모 박스 설정 (이미지의 크기에 따라 네모 박스 크기 조절 가능)
    height, width, _ = image.shape
    box_start = (50, 50)
    box_end = (width - 50, height - 50)
    
    # 네모 박스 그리기 (색상: 빨강, 두께: 2)
    cv2.rectangle(image, box_start, box_end, (0, 0, 255), 2)
    
    # 라벨 텍스트 추가
    cv2.putText(image, label, (box_start[0], box_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 이미지를 출력 (저장하지 않고 바로 화면에 표시)
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)  # 아무 키나 누를 때까지 대기
    cv2.destroyAllWindows()

# 테스트 함수
def test_model(model, image_path, transform, class_names, device):
    # 이미지 예측
    predicted_class_idx = predict_image(model, image_path, transform, device)
    predicted_class = class_names[predicted_class_idx]

    # 네모 박스 그리기 및 출력
    draw_bounding_box(image_path, predicted_class)

# 메인 실행 부분
if __name__ == "__main__":
    # 파라미터 설정
    model_path = "data/models/best_crnn_model.pth"  # 모델 경로
    image_path = "1.jpg"  # 테스트할 이미지 경로
    n_classes = 5  # 클래스 수 (실제 데이터에 맞게 설정)
    
    # 학습 시 사용한 데이터 변환기와 동일한 변환기 정의
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 클래스 이름 리스트 (실제 데이터셋에 맞게 수정)
    class_names = ["SUV", "버스", "세단", "이륜차", "트럭"]
    
    # 장치 설정 (GPU 사용 가능 여부 확인)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = load_model(model_path, n_classes).to(device)

    # 테스트 실행
    test_model(model, image_path, transform, class_names, device)
