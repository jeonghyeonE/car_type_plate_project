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
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler  # WeightedRandomSampler를 위한 import
import numpy as np

# 1. 데이터셋 클래스 정의
class VehicleDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, mode='train', max_images=None, type_images=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.mode = mode
        self.max_images = max_images
        self.type_images = type_images

        # 이미지 경로와 라벨을 불러오기
        for vehicle_type in os.listdir(img_dir):
            image_count = 0
            if vehicle_type != 'SUV' and vehicle_type != '버스' and vehicle_type != '세단'and vehicle_type != '이륜차' and vehicle_type != '트럭':
                continue
            print(f"{self.mode} - {vehicle_type}")
            if vehicle_type == '라벨링데이터':
                continue
            vehicle_path = os.path.join(img_dir, vehicle_type)
            for model in os.listdir(vehicle_path):
                model_path = os.path.join(vehicle_path, model)
                type_count = 0
                for img_file in os.listdir(model_path):
                    if image_count >= self.max_images or (type_count >= type_images and (vehicle_type == 'SUV' or vehicle_type == '세단') and mode != 'val'):
                        break
                    img_path = os.path.join(model_path, img_file)
                    json_path = os.path.join(label_dir, vehicle_type, model, img_file.replace('.jpg', '.json'))
                    
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                            self.image_paths.append(img_path)
                            self.labels.append(label_data['car']['attributes']['model'].split('_')[0])
                            image_count += 1
                            type_count += 1
                if image_count >= self.max_images:
                    break
        
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

# 2. CRNN 모델 정의
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
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        
        # RNN에 입력을 맞추기 위해 크기 조정
        conv = conv.squeeze(2)  # (batch_size, channels, width)
        conv = conv.permute(0, 2, 1)  # (batch_size, width, channels)
        
        output, _ = self.rnn(conv)
        
        # 시퀀스의 마지막 출력만 사용하여 분류
        output = output[:, -1, :]  # (batch_size, nh * 2)
        output = self.fc(output)  # (batch_size, nclass)
        
        return output

# 3. 학습 함수 정의 (Gradient Clipping 추가)
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
        
        # Gradient Clipping 적용
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy

# 4. 검증 함수 정의
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

# 5. 학습 및 검증 데이터셋 및 데이터로더 정의 (Data Augmentation 및 WeightedRandomSampler 적용)
# 데이터 보강을 위한 transform 정의
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 크기 조정 (여기서 비율을 유지한 크기로 변경 가능)
    transforms.RandomRotation(15),  # 최대 15도 회전
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 10% 정도 이동
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 밝기, 대비, 채도, 색조 변화
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지 정규화
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
    max_images=10000,
    type_images=600
)

val_dataset = VehicleDataset(
    img_dir="D:/dataset/자동차_차종_번호판_데이터/Validation", 
    label_dir="D:/dataset/자동차_차종_번호판_데이터/Validation/라벨링데이터/차종분류데이터", 
    transform=val_transform, 
    mode='val', 
    max_images=1000,
    type_images=100
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
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 6. 학습에 필요한 요소 정의 (Early Stopping 및 Learning Rate Scheduler 추가)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(imgH=128, nc=3, nclass=5, nh=256).to(device) # nclass=(train_dataset.labels.max()+1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler 정의
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# EarlyStopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# EarlyStopping 객체 생성
early_stopping = EarlyStopping(patience=10, delta=0.01)

# 7. 데이터셋 클래스별 데이터 개수 출력
train_class_distribution = train_dataset.get_class_distribution()
val_class_distribution = val_dataset.get_class_distribution()

print("Training set class distribution:")
for class_name, count in train_class_distribution.items():
    print(f"{class_name}: {count} samples")

print("\nValidation set class distribution:")
for class_name, count in val_class_distribution.items():
    print(f"{class_name}: {count} samples")

# 8. 학습 및 검증 진행
best_val_acc = 0  # 최고 검증 정확도 초기값
save_path = 'data/models/best_crnn_model.pth'  # 모델을 저장할 경로

for epoch in range(30):
    print(f"\nEpoch {epoch+1}/30")
    
    # 학습
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    
    # 검증
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    # Learning Rate Scheduler 업데이트
    scheduler.step(val_loss)
    
    # 모델 저장: 검증 정확도가 최고일 때 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%")
    
    # Early Stopping 체크
    early_stopping(val_acc)
    if early_stopping.early_stop:
        print("Early stopping")
        break
