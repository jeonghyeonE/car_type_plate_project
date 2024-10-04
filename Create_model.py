import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler  # WeightedRandomSampler를 위한 import
from collections import Counter
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib import rc
import torchvision.models as models

# 한글 폰트 설정
font_path = 'C:/practice_coding/data/NanumGothic.ttf'  # 한글 폰트 경로를 지정
fontprop = fm.FontProperties(fname=font_path)
rc('font', family=fontprop.get_name())

# 한글이 포함된 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 랜덤 시드 고정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대해 시드 고정
    torch.backends.cudnn.deterministic = True  # CUDNN에서 결정론적 동작
    torch.backends.cudnn.benchmark = False  # 성능 저하 가능성 있지만 재현성을 위해 필요

# 시드 고정 (예: 42)
set_seed(42)

# 데이터셋 클래스 정의
class VehicleDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, mode='train', max_images_per_class=None, max_images_per_brand=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.mode = mode
        self.max_images_per_class = max_images_per_class  # 클래스별 최대 이미지 수
        self.max_images_per_brand = max_images_per_brand  # 브랜드별 최대 이미지 수

        # 이미지 경로와 라벨을 불러오기
        for vehicle_type in os.listdir(img_dir):
            if vehicle_type != 'SUV' and vehicle_type != '버스' and vehicle_type != '세단' and vehicle_type != '이륜차' and vehicle_type != '트럭':
                continue  # 지정된 클래스가 아닐 경우 패스
            
            print(f"{self.mode} - {vehicle_type}")
            
            vehicle_path = os.path.join(img_dir, vehicle_type)
            class_image_count = 0  # 클래스별 이미지 개수 카운트
            
            for model in os.listdir(vehicle_path):
                model_path = os.path.join(vehicle_path, model)
                
                # 이미지 파일을 랜덤하게 섞음
                image_files = os.listdir(model_path)
                random.shuffle(image_files)  # 이미지를 랜덤하게 섞음

                brand_image_count = 0  # 브랜드별 이미지 개수 카운트
                
                for img_file in image_files:
                    if class_image_count >= self.max_images_per_class:
                        break  # 클래스별 최대 이미지 개수를 넘으면 중단
                    if vehicle_type == 'SUV':
                        if mode == 'val':
                            if brand_image_count >= self.max_images_per_brand/5:
                                break  # 브랜드별 최대 이미지 개수를 넘으면 중단
                        else:
                            if brand_image_count >= self.max_images_per_brand/18:
                                break  # 브랜드별 최대 이미지 개수를 넘으면 중단
                    elif vehicle_type == '세단':
                        if mode == 'val':
                            if brand_image_count >= self.max_images_per_brand/7:
                                break  # 브랜드별 최대 이미지 개수를 넘으면 중단
                        else:
                            if brand_image_count >= self.max_images_per_brand/14:
                                break  # 브랜드별 최대 이미지 개수를 넘으면 중단
                    else:
                        if brand_image_count >= self.max_images_per_brand:
                            break  # 브랜드별 최대 이미지 개수를 넘으면 중단

                    img_path = os.path.join(model_path, img_file)
                    json_path = os.path.join(label_dir, vehicle_type, model, img_file.replace('.jpg', '.json'))
                    
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                            self.image_paths.append(img_path)
                            self.labels.append(label_data['car']['attributes']['model'].split('_')[0])
                            class_image_count += 1  # 클래스 이미지 개수 업데이트
                            brand_image_count += 1  # 브랜드 이미지 개수 업데이트

                if class_image_count >= self.max_images_per_class:
                    break  # 클래스별 최대 이미지 개수를 넘으면 중단

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
        
        # 라벨을 텐서로 변환 (정수형 라벨)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def get_class_distribution(self):
        """
        클래스별 데이터 수를 출력하는 함수
        """
        label_counts = Counter(self.labels)
        classes = self.label_encoder.classes_
        class_distribution = {classes[label]: count for label, count in label_counts.items()}
        return class_distribution

class ModifiedResNet50Model(nn.Module):
    def __init__(self, nclass, pretrained=True):
        super(ModifiedResNet50Model, self).__init__()
        
        # torchvision에서 제공하는 ResNet50 모델 불러오기
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # 기존 FC 레이어를 덮어쓰기 전에 새로운 FC 레이어를 추가
        num_ftrs = self.resnet50.fc.in_features  # ResNet50의 기본 FC 입력 차원
        
        # 새로운 FC 레이어 추가
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # 첫 번째 FC 레이어 (새로 추가된 레이어)
            nn.ReLU(True),             # 활성화 함수
            nn.Dropout(0.5),           # 드롭아웃으로 과적합 방지
            
            nn.Linear(512, nclass)     # 최종 출력 레이어 (nclass로 설정)
        )
    
    def forward(self, x):
        return self.resnet50(x)


# 학습 함수 정의 (Gradient Clipping 삭제)
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient Clipping 추가
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy

# 검증 함수 정의
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy

# 데이터 보강을 위한 transform 정의
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # transforms.RandomRotation(15),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지 정규화
])

# 데이터셋 생성
train_dataset = VehicleDataset(
    img_dir="D:/dataset/자동차_차종_번호판_데이터/Training", 
    label_dir="D:/dataset/자동차_차종_번호판_데이터/Training/라벨링데이터/차종분류데이터", 
    transform=train_transform, 
    mode='train', 
    max_images_per_class=30000,   # 각 클래스별 최대 1000장
    max_images_per_brand=6000     # 각 브랜드별 최대 300장
)

val_dataset = VehicleDataset(
    img_dir="D:/dataset/자동차_차종_번호판_데이터/Validation", 
    label_dir="D:/dataset/자동차_차종_번호판_데이터/Validation/라벨링데이터/차종분류데이터", 
    transform=val_transform, 
    mode='val', 
    max_images_per_class=10000,   # 각 클래스별 최대 1000장
    max_images_per_brand=600     # 각 브랜드별 최대 300장
)

# 클래스별 샘플 수 계산
class_sample_count = np.array([len(np.where(train_dataset.labels == t)[0]) for t in np.unique(train_dataset.labels)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in train_dataset.labels])

samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()

# WeightedRandomSampler 생성
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# 데이터로더 생성 (sampler 사용)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 학습에 필요한 요소 정의 (Early Stopping 및 Learning Rate Scheduler 추가)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model = ModifiedResNet50Model(nclass=5).to(device)  # 클래스 수는 5로 설정
criterion = nn.CrossEntropyLoss()  # Categorical Crossentropy에 대응
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning Rate Scheduler 정의
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=5, delta=0.01, mode='val_acc'):
        """
        Args:
            patience (int): 성능이 개선되지 않을 때 기다릴 에포크 수
            delta (float): 성능 향상의 최소 기준
            mode (str): 'val_loss' 또는 'val_acc' 중 하나를 선택하여 기준을 설정
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode  # 'val_loss' 또는 'val_acc'를 사용하여 early stopping 기준 설정
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_metric):
        """
        Args:
            val_metric (float): val_loss 또는 val_acc 값
        """
        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'val_acc':
            # 정확도를 사용할 경우 성능이 좋아지지 않으면 카운터 증가
            if score < self.best_score + self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == 'val_loss':
            # 손실을 사용할 경우 성능이 나빠지면 카운터 증가
            if score > self.best_score - self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

# EarlyStopping 객체 생성
early_stopping = EarlyStopping(patience=10, delta=0.01, mode='val_loss')

# 데이터셋 클래스별 데이터 개수 출력
train_class_distribution = train_dataset.get_class_distribution()
val_class_distribution = val_dataset.get_class_distribution()

print("Training set class distribution:")
for class_name, count in train_class_distribution.items():
    print(f"{class_name}: {count} samples")

print("\nValidation set class distribution:")
for class_name, count in val_class_distribution.items():
    print(f"{class_name}: {count} samples")

# 학습 및 검증 진행
best_val_acc = 0  # 최고 검증 정확도 초기값
best_val_loss = float('inf')  # 최고 검증 손실 초기값 (초기값은 무한대 설정)
save_path = 'data/models/best_cnn_model.pth'  # 모델을 저장할 경로

# 학습 및 검증 기록을 저장할 리스트
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

epoch_num = 100

for epoch in range(epoch_num):
    print(f"\nEpoch {epoch+1}/{epoch_num}")
    
    # 학습
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    
    # 검증
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # 학습 및 검증 결과 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    # Learning Rate Scheduler 업데이트
    scheduler.step(val_loss)
    
    # 모델 저장: 검증 정확도가 최고일 때 모델 저장
    if early_stopping.mode == 'val_acc' and val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%")
    
    elif early_stopping.mode == 'val_loss' and val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    
    # Early Stopping 체크
    if early_stopping.mode == 'val_acc':
        early_stopping(val_acc)
    elif early_stopping.mode == 'val_loss':
        early_stopping(val_loss)
        
    if early_stopping.early_stop:
        print("Early stopping")
        break

# 학습 손실 및 정확도 시각화
def plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 손실 그래프
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

# 학습 곡선 시각화
# plot_learning_curve(train_losses, val_losses, train_accuracies, val_accuracies)

# 검증 데이터에 대해 예측 및 컨퓨전 매트릭스 계산
def compute_confusion_matrix(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 컨퓨전 매트릭스 계산
    cm = confusion_matrix(all_labels, all_preds)

    # 컨퓨전 매트릭스 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('컨퓨전 매트릭스', fontproperties=fontprop)  # 한글 제목
    plt.xlabel('예측된 라벨', fontproperties=fontprop)  # 한글 X축
    plt.ylabel('실제 라벨', fontproperties=fontprop)    # 한글 Y축
    plt.show()


    # 클래스 이름 정의 (예: ['SUV', '세단', '버스', '이륜차', '트럭'])
class_names = train_dataset.label_encoder.classes_

# 컨퓨전 매트릭스 계산 및 시각화
# compute_confusion_matrix(model, val_loader, device, class_names)