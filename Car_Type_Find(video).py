import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.models as models

# PyTorch Hub에서 YOLOv5 모델 로드
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# EfficientNetB0 모델로 수정
class ModifiedEfficientNetB0Model(nn.Module):
    def __init__(self, nclass, pretrained=True):
        super(ModifiedEfficientNetB0Model, self).__init__()
        
        # torchvision에서 제공하는 EfficientNetB0 모델 불러오기
        self.efficientnet_b0 = models.efficientnet_b0(pretrained=pretrained)
        
        # 기존 FC 레이어를 덮어쓰기 전에 새로운 FC 레이어를 추가
        num_ftrs = self.efficientnet_b0.classifier[1].in_features  # EfficientNetB0의 기본 FC 입력 차원
        
        # 새로운 FC 레이어 추가
        self.efficientnet_b0.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # 첫 번째 FC 레이어 (새로 추가된 레이어)
            nn.ReLU(True),             # 활성화 함수
            nn.Dropout(0.5),           # 드롭아웃으로 과적합 방지
            nn.Linear(512, nclass)     # 최종 출력 레이어 (nclass로 설정)
        )
    
    def forward(self, x):
        return self.efficientnet_b0(x)

# 차량 클래스 정의
vehicle_classes = ['SUV', '버스', '세단', '이륜차', '트럭']

# 사전 학습된 ModifiedResNet50Model 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ModifiedEfficientNetB0Model(nclass=5).to(device)
model.load_state_dict(torch.load('data/models/best_model.pth'))  # 훈련된 모델 경로
model.eval()

# 이미지 전처리 변환 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 차량 분류 함수
def classify_vehicle(image, model, transform, device):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)  # 이미지 전처리
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
    return vehicle_classes[predicted.item()]

# 한글 텍스트를 영상에 추가하는 함수
def put_korean_text(image, text, position, font_path='data/NanumGothic.ttf', font_size=30, color=(255, 0, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 추가
    draw.text(position, text, font=font, fill=color)

    # PIL 이미지를 다시 OpenCV 이미지로 변환
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# CCTV 영상 처리
video_path = 'abcd.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 차량 탐지 (YOLOv5)
    results = yolo_model(frame)

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = yolo_model.names[int(cls)]
        
        # 차량과 관련된 클래스만 처리 (자동차, 버스, 트럭 등)
        if label in ['car', 'bus', 'truck', 'motorbike']:
            # 탐지된 차량 부분만 크롭
            vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # 차량 종류 분류 (CRNN)
            vehicle_type = classify_vehicle(vehicle_img, model, transform, device)
            
            # 결과를 영상에 표시 (한글 폰트 사용)
            frame = put_korean_text(frame, f'{vehicle_type}', (int(x1), int(y1) - 40))

            # 탐지 박스 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # 영상 출력
    cv2.imshow('Vehicle Classification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
