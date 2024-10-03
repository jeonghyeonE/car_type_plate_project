import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# PyTorch Hub에서 YOLOv5 모델 로드
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# CRNN 모델 정의 (이전 코드 그대로 사용)
class CRNN(torch.nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.AdaptiveAvgPool2d((1, None))
        )

        self.rnn = torch.nn.LSTM(256, nh, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        output, _ = self.rnn(conv)
        output = output[:, -1, :]
        output = self.fc(output)
        return output

# 차량 클래스 정의
vehicle_classes = ['SUV', '버스', '세단', '승합', '이륜차', '트럭', '해치백', '화물']

# 사전 학습된 CRNN 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
crnn_model = CRNN(imgH=128, nc=3, nclass=8, nh=256).to(device)
crnn_model.load_state_dict(torch.load('data/models/best_crnn_model.pth'))  # 훈련된 모델 경로
crnn_model.eval()

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
video_path = 'data/test2.mp4'
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
            vehicle_type = classify_vehicle(vehicle_img, crnn_model, transform, device)
            
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
