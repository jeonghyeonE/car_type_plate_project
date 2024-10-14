import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.models as models
from ultralytics import YOLO  # YOLOv8 라이브러리 임포트

is_paused = False

# YOLOv8 번호판 감지 모델 로드 (학습된 모델 경로 설정)
plate_model_path = 'runs/detect/plate_detection4/weights/best.pt'
plate_model = YOLO(plate_model_path)  # YOLOv8 custom model

# YOLOv8 차량 감지 모델 로드 (사전 학습된 모델 사용)
vehicle_model = YOLO('yolov8s.pt')  # YOLOv8 사전 학습된 모델 (n 버전)

# EfficientNetB0 모델로 수정 (기존 코드 유지)
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

# 사전 학습된 차량 분류 모델 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ModifiedEfficientNetB0Model(nclass=5).to(device)
model.load_state_dict(torch.load('data/models/best_model.pth'))  # 훈련된 모델 경로
model.eval()

# 이미지 전처리 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
# video_path = 'opencvDojang/data/carplate2.mp4'
video_path = '17s.mp4'
cap = cv2.VideoCapture(video_path)

# 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (mp4v 사용)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 원본 영상의 FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 원본 영상의 폭
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 영상의 높이

# 저장할 동영상 파일 초기화
output_path = 'output_detected_video.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 차량 탐지 (YOLOv8 차량 탐지 모델 사용)
    vehicle_results = vehicle_model(frame)

    # 번호판 탐지 (YOLOv8 번호판 감지 모델 사용)
    plate_results = plate_model(frame)

    # 차량 탐지 결과 처리
    for result in vehicle_results[0].boxes:  # YOLOv8은 'boxes'로 탐지 결과 접근
        x1, y1, x2, y2 = result.xyxy[0].tolist()  # 탐지 박스 좌표
        conf = result.conf.tolist()[0]  # 신뢰도
        cls = int(result.cls.tolist()[0])  # 클래스 인덱스
        label = vehicle_model.names[cls]  # 클래스 이름
        
        # 차량과 관련된 클래스만 처리 (자동차, 버스, 트럭 등)
        if label in ['car', 'bus', 'truck', 'motorbike']:
            # 탐지된 차량 부분만 크롭
            vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # 차량 종류 분류 (EfficientNetB0 모델 사용)
            vehicle_type = classify_vehicle(vehicle_img, model, transform, device)
            
            # 차량 종류 결과를 영상에 표시 (한글 폰트 사용)
            frame = put_korean_text(frame, f'{vehicle_type}', (int(x1), int(y1) - 40))

            # 탐지 박스 그리기 (차량)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # 번호판 탐지 결과 처리
    for plate_result in plate_results[0].boxes:
        px1, py1, px2, py2 = plate_result.xyxy[0].tolist()  # 탐지 박스 좌표
        pconf = plate_result.conf.tolist()[0]  # 신뢰도
        pcls = int(plate_result.cls.tolist()[0])  # 클래스 인덱스
        
        # 번호판 탐지 박스 그리기
        cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
        # frame = put_korean_text(frame, '', (int(px1), int(py1) - 40), color=(0, 255, 255))
        frame = put_korean_text(frame, plate_model.names[pcls], (int(px1), int(py1) - 40), color=(0, 255, 255))

    # 영상 출력
    cv2.imshow('Vehicle and License Plate Detection', frame)

    # 저장된 프레임을 비디오 파일에 기록
    out.write(frame)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    key = cv2.waitKey(0 if is_paused else 1)  # 키 입력 대기s
    if key == 27:  # ESC 키를 누르면 종료
        break
    elif key == ord('s'):  # s 키를 누르면 일시 정지/재생 토글
        is_paused = not is_paused

cap.release()
out.release()  # 저장 비디오 파일 닫기
cv2.destroyAllWindows()
