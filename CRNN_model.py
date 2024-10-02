import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # tqdm 추가
from collections import Counter  # 클래스별 데이터 수를 확인하기 위한 Counter 추가

# 1. 데이터셋 클래스 정의
class VehicleDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, mode='train', max_images=5000):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()  # 라벨 인코더 초기화
        self.mode = mode  # train 또는 val 모드
        self.max_images = max_images  # 이미지 개수 제한

        # 이미지 경로와 라벨을 불러오기
        for vehicle_type in os.listdir(img_dir):
            image_count = 0  # 저장된 이미지 개수를 셀 변수를 추가합니다.
            print(f"{self.mode} - {vehicle_type}")
            if vehicle_type == '라벨링데이터':
                continue
            vehicle_path = os.path.join(img_dir, vehicle_type)
            for model in os.listdir(vehicle_path):
                model_path = os.path.join(vehicle_path, model)
                for img_file in os.listdir(model_path):
                    if image_count >= self.max_images:  # 이미지 개수 제한
                        break
                    img_path = os.path.join(model_path, img_file)
                    json_path = os.path.join(label_dir, vehicle_type, model, img_file.replace('.jpg', '.json'))
                    
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)
                            self.image_paths.append(img_path)
                            self.labels.append(label_data['car']['attributes']['model'].split('_')[0])
                            image_count += 1  # 이미지가 추가될 때마다 카운트 증가
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

# 2. CRNN 모델 정의 (변경 없음)
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

# 3. 학습 함수 정의 (tqdm 추가)
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # tqdm을 사용하여 학습 진행 상황 표시
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy

# 4. 검증 함수 정의 (tqdm 추가)
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # tqdm을 사용하여 검증 진행 상황 표시
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

# 5. 학습 및 검증 데이터셋 및 데이터로더 정의
train_dataset = VehicleDataset(
    img_dir="D:/dataset/자동차_차종_번호판_데이터/Training", 
    label_dir="D:/dataset/자동차_차종_번호판_데이터/Training/라벨링데이터/차종분류데이터", 
    transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]), 
    mode='train', 
    max_images=5000  # train에서는 5000장의 이미지로 제한
)

val_dataset = VehicleDataset(
    img_dir="D:/dataset/자동차_차종_번호판_데이터/Validation", 
    label_dir="D:/dataset/자동차_차종_번호판_데이터/Validation/라벨링데이터/차종분류데이터", 
    transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]), 
    mode='val', 
    max_images=1000  # val에서는 1000장의 이미지로 제한
)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 6. 학습 및 검증 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(imgH=128, nc=3, nclass=(train_dataset.labels.max()+1), nh=256).to(device) # model = CRNN(imgH=128, nc=3, nclass=len(train_dataset.labels), nh=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    print(f"Epoch {epoch+1}/30")
    
    # 학습
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    
    # 검증
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}%")
    
    # 모델 저장: 검증 정확도가 최고일 때 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with validation accuracy: {val_acc:.4f}%")
