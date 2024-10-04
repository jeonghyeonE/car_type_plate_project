import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
import random

# 한글 폰트 설정
font_path = 'C:/practice_coding/data/NanumGothic.ttf'  # 시스템에 설치된 폰트 경로를 지정 (예: NanumGothic)
fontprop = fm.FontProperties(fname=font_path)
rc('font', family=fontprop.get_name())

# 학습에 사용된 CRNN 모델 정의 (이전에 정의한 CRNN 모델과 동일하게 유지)
class CNN(nn.Module):
    def __init__(self, imgH, nc, nclass):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=3, stride=1, padding='same'),  # 첫 번째 Conv2D: 필터 32개, padding='same'
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # MaxPooling: 이미지 크기 반으로 줄임
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),  # 두 번째 Conv2D: 필터 64개, padding='same'
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # MaxPooling: 이미지 크기 반으로 줄임
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),  # 세 번째 Conv2D: 필터 128개
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Flatten(),  # Flatten을 통해 1D 텐서로 변환
            
            nn.Linear(16*16*128, 512),  # Fully connected layer
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(512, nclass),  # 최종 출력: 클래스 수(nclass)
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.cnn(x)

# 데이터셋 클래스 (각 클래스의 마지막 100장씩만 선택)
class VehicleDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, mode='test', max_images_per_class=100):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.mode = mode
        self.max_images_per_class = max_images_per_class

        # 이미지 경로와 라벨을 불러오기
        for vehicle_type in os.listdir(img_dir):
            if vehicle_type not in ['SUV', '버스', '세단', '이륜차', '트럭']:
                continue
            image_paths = []  # 각 vehicle_type에 해당하는 이미지 경로를 담을 리스트
            vehicle_path = os.path.join(img_dir, vehicle_type)
            for model in os.listdir(vehicle_path):
                model_path = os.path.join(vehicle_path, model)
                for img_file in os.listdir(model_path):
                    img_path = os.path.join(model_path, img_file)
                    json_path = os.path.join(label_dir, vehicle_type, model, img_file.replace('.jpg', '.json'))

                    if os.path.exists(json_path):
                        image_paths.append((img_path, json_path))  # 이미지 경로와 json 경로를 튜플로 저장
            
            # 랜덤으로 100장의 이미지를 선택 (각 vehicle_type별)
            selected_paths = random.sample(image_paths, min(len(image_paths), self.max_images_per_class))
            for img_path, json_path in selected_paths:
                with open(json_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    self.image_paths.append(img_path)
                    self.labels.append(label_data['car']['attributes']['model'].split('_')[0])

        # 라벨을 정수형으로 변환
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# 모델 테스트 함수 정의 (눈으로 확인하는 기능 포함)
def test_model_with_visualization(model_path, test_loader, device):
    # 모델 초기화 및 가중치 로드
    model = CNN(imgH=128, nc=3, nclass=5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    class_names = test_loader.dataset.label_encoder.classes_

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            class_probs = torch.softmax(outputs, dim=1)  # 각 클래스에 대한 확률 계산
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 눈으로 확인 (테스트 데이터 중 일부만 표시)
            visualize_predictions(images.cpu(), labels.cpu(), predicted.cpu(), class_probs.cpu(), class_names)

    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 예측 결과를 시각화하는 함수
def visualize_predictions(images, labels, predictions, class_probs, class_names, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        image = images[i].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        label = class_names[labels[i]]
        predicted_label = class_names[predictions[i]]
        probs = class_probs[i].cpu().numpy() * 100  # 각 클래스에 대한 확률값 (%) 계산

        # 모든 클래스 확률 출력
        prob_text = "\n".join([f"{class_names[j]}: {probs[j]:.2f}%" for j in range(len(class_names))])

        axes[i].imshow(image)
        axes[i].set_title(f"True: {label}\nPred: {predicted_label}", fontproperties=fontprop)
        axes[i].text(0, -30, prob_text, fontsize=9, fontproperties=fontprop)  # 각 클래스의 확률 출력
        axes[i].axis('off')

    plt.show()

# 테스트용 데이터셋 및 데이터로더 정의
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 데이터셋 경로 설정
test_dataset = VehicleDataset(
    img_dir="D:/dataset/자동차_차종_번호판_데이터/Validation", 
    label_dir="D:/dataset/자동차_차종_번호판_데이터/Validation/라벨링데이터/차종분류데이터", 
    transform=test_transform, 
    mode='test', 
    max_images_per_class=100  # 각 클래스의 마지막 100장 사용
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 테스트 코드 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'data/models/best_cnn_model.pth'  # 학습된 모델 경로
    test_model_with_visualization(model_path, test_loader, device)
