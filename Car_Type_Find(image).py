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

# 사전 학습된 CRNN 모델 로드
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
        probs = torch.softmax(output, dim=1)  # 모든 클래스에 대한 확률 계산
        _, predicted = output.max(1)
        predicted_class = predicted.item()
        predicted_prob = probs[0, predicted_class].item() * 100  # 예측된 클래스의 확률 (%) 반환

        # 모든 클래스의 확률을 반환
        class_probs = probs[0].cpu().numpy() * 100
    return vehicle_classes[predicted_class], predicted_prob, class_probs

# 한글 텍스트를 이미지에 추가하는 함수
def put_korean_text(image, text, position, font_path='data/NanumGothic.ttf', font_size=20, color=(255, 0, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 추가
    draw.text(position, text, font=font, fill=color)

    # PIL 이미지를 다시 OpenCV 이미지로 변환
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# 테스트할 이미지 경로
image_path = 'data/1.jpg'

# 이미지 읽기
image = cv2.imread(image_path)

# 차량 탐지 (YOLOv5)
results = yolo_model(image)

for result in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = result
    label = yolo_model.names[int(cls)]
    
    # 차량과 관련된 클래스만 처리 (자동차, 버스, 트럭 등)
    if label in ['car', 'bus', 'truck', 'motorbike']:
        # 탐지된 차량 부분만 크롭
        vehicle_img = image[int(y1):int(y2), int(x1):int(x2)]
        
        # 차량 종류 분류 (CRNN) 및 확률 계산
        vehicle_type, vehicle_prob, class_probs = classify_vehicle(vehicle_img, model, transform, device)
        
        # 모든 클래스 확률을 텍스트로 생성
        class_prob_text = "\n".join([f"{vehicle_classes[i]}: {class_probs[i]:.2f}%" for i in range(len(vehicle_classes))])
        
        # 결과를 이미지에 표시 (한글 폰트 사용)
        # display_text = f'{vehicle_type} ({vehicle_prob:.2f}%)\n{class_prob_text}'
        display_text = f'{vehicle_type} ({vehicle_prob:.2f}%)'
        image = put_korean_text(image, display_text, (int(x1), int(y1) - 40))  # 텍스트 위치 조정

        # 탐지 박스 그리기
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# 이미지 출력
cv2.imshow('Vehicle Classification', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
