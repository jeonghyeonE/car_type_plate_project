import albumentations as A
import cv2
import os

# YOLO 형식의 bounding box 변환 함수 (x_center, y_center, width, height -> x_min, y_min, x_max, y_max)
# YOLO 형식의 bounding box 좌표를 Pascal VOC 형식으로 변환합니다. 
# YOLO 좌표는 이미지의 중앙을 기준으로 상대적인 위치를 나타내며, Pascal VOC 좌표는 절대적인 픽셀 값을 나타냅니다.
def yolo_to_pascal_voc(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * img_width  # x 중심에서 너비 절반을 뺀 좌표
    y_min = (y_center - height / 2) * img_height  # y 중심에서 높이 절반을 뺀 좌표
    x_max = (x_center + width / 2) * img_width  # x 중심에서 너비 절반을 더한 좌표
    y_max = (y_center + height / 2) * img_height  # y 중심에서 높이 절반을 더한 좌표
    return [x_min, y_min, x_max, y_max]

# Pascal VOC 형식을 YOLO 형식으로 변환하는 함수 (x_min, y_min, x_max, y_max -> x_center, y_center, width, height)
# Pascal VOC 형식의 bounding box 좌표를 YOLO 형식으로 변환합니다. 
# Pascal VOC 좌표는 절대적인 픽셀 값을 사용하며, YOLO 좌표는 이미지 중앙을 기준으로 상대적인 위치를 사용합니다.
def pascal_voc_to_yolo(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_width  # x 좌표의 중간 값을 YOLO의 x 중심 좌표로 변환
    y_center = (y_min + y_max) / 2 / img_height  # y 좌표의 중간 값을 YOLO의 y 중심 좌표로 변환
    width = (x_max - x_min) / img_width  # 폭을 YOLO 형식의 상대적인 값으로 변환
    height = (y_max - y_min) / img_height  # 높이를 YOLO 형식의 상대적인 값으로 변환
    return [x_center, y_center, width, height]

# 증강 방법을 파일명에 추가하는 함수 (Replay 정보에서 클래스 이름을 추출)
# Albumentations의 Replay 기능에서 사용된 증강 방법 이름을 추출해 파일명에 추가합니다.
def get_augmentation_name(augmentation_applied):
    names = []
    if isinstance(augmentation_applied, list):
        for aug in augmentation_applied:
            if '__class_fullname__' in aug:
                aug_name = aug['__class_fullname__']  # 증강 방법의 클래스 이름 추출
                names.append(aug_name)
    return '_'.join(names)  # 여러 증강 방법이 적용되면, 이름을 "_"로 연결해 반환

# 모든 증강 방법을 정의하고 선택된 증강 방법만 적용하는 함수
# 사용자 선택에 따라 원하는 증강 방법을 적용할 수 있도록 모든 증강 기법을 제공하는 딕셔너리를 사용합니다.
def get_all_transforms(selected_augmentations):
    # Albumentations 라이브러리의 다양한 증강 기법을 딕셔너리 형태로 정의합니다.
    augmentations = {
        'HorizontalFlip': A.HorizontalFlip(p=1.0),  # 이미지의 좌우를 뒤집습니다.
        'VerticalFlip': A.VerticalFlip(p=1.0),  # 이미지의 상하를 뒤집습니다.
        'ShiftScaleRotate': A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1.0),  
        # 이미지를 이동, 확대/축소, 회전하는 변환입니다.
        'RandomBrightnessContrast': A.RandomBrightnessContrast(p=1.0),  # 이미지의 밝기와 대비를 무작위로 변경합니다.
        'MotionBlur': A.MotionBlur(p=1.0),  # 이미지에 모션 블러 효과를 적용합니다.
        'GaussianBlur': A.GaussianBlur(p=1.0),  # 이미지에 가우시안 블러 효과를 적용합니다.
        'MedianBlur': A.MedianBlur(blur_limit=3, p=1.0),  # 이미지에 중앙값 블러 효과를 적용합니다.
        'RandomRain': A.RandomRain(p=1.0),  # 이미지에 무작위로 비 효과를 추가합니다.
        'RandomSnow': A.RandomSnow(p=1.0),  # 이미지에 무작위로 눈 효과를 추가합니다.
        'RandomFog': A.RandomFog(p=1.0),  # 이미지에 무작위로 안개 효과를 추가합니다.
        'CLAHE': A.CLAHE(p=1.0),  # CLAHE(대비 제한 적응 히스토그램 평활화) 필터를 적용하여 대비를 높입니다.
        'RandomGamma': A.RandomGamma(p=1.0),  # 감마 조정을 무작위로 적용합니다.
        'HueSaturationValue': A.HueSaturationValue(p=1.0),  # 색조, 채도, 밝기를 무작위로 조정합니다.
        'RandomShadow': A.RandomShadow(p=1.0),  # 이미지에 그림자를 무작위로 추가합니다.
        'Solarize': A.Solarize(p=1.0),  # Solarize 효과를 적용하여 밝은 픽셀을 반전시킵니다.
        'Posterize': A.Posterize(p=1.0),  # 이미지의 색상 깊이를 줄입니다.
        'Equalize': A.Equalize(p=1.0),  # 히스토그램 평활화를 통해 이미지를 균일하게 만듭니다.
        'Perspective': A.Perspective(scale=(0.5, 0.1), p=1.0), # 각도나 거리 변화에 따른 원근감을 반영하여 데이터를 증강할 수 있습니다.
        'Resize': A.Resize(height=int(244*0.5), width=int(244*0.5)), # 이미지를 축소하는 증강 기법입니다. 
        'Affine': A.Affine(scale=(0.8, 1.2), rotate=(-10, 10), shear=(-5, 5), p=1.0) # 위치와 각도를 변경하는 방식으로 데이터를 증강할 수 있습니다.
    }
    
    # 선택된 증강 기법만 추출하여 리스트로 반환합니다.
    selected_transforms = [augmentations[aug] for aug in selected_augmentations if aug in augmentations]
    
    # 선택된 증강 기법들이 담긴 Albumentations Compose 객체를 반환합니다.
    return A.Compose(selected_transforms)


# 다양한 스케일로 증강을 시도하는 함수
# ShiftScaleRotate 변환에서 다양한 scale 값을 사용해 증강을 적용하며, 적용된 증강 방법에 따라 이미지와 라벨을 저장합니다.
def augment_with_different_scales(image, bboxes, class_ids, img_width, img_height, base_name, scale_limits, augmented_image_dir, augmented_label_dir):
    for scale in scale_limits:

        # ShiftScaleRotate 증강 기법에 대해 다양한 스케일 값을 적용
        transform = A.ReplayCompose(
            [A.ShiftScaleRotate(shift_limit=0.1, scale_limit=scale, rotate_limit=30, p=1.0)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'])
            )

        # 증강 적용
        augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
        augmentation_applied = augmented['replay']['transforms']  # Replay 정보 사용 (적용된 증강 방법 정보 저장)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_ids = augmented['class_ids']

        # 유효한 bounding box만 선택 (유효하지 않은 좌표는 필터링)
        valid_bboxes = []
        valid_class_ids = []
        for bbox, class_id in zip(augmented_bboxes, augmented_class_ids):
            x_min, y_min, x_max, y_max = bbox
            # 이미지 범위를 벗어나지 않는 유효한 박스만 유지
            if x_min >= 0 and y_min >= 0 and x_max <= img_width and y_max <= img_height and x_min < x_max and y_min < y_max:
                valid_bboxes.append(bbox)
                valid_class_ids.append(class_id)

        # 유효한 bounding box가 없으면 스킵
        if not valid_bboxes:
            continue

        # 증강 방법 이름 추출
        aug_name = get_augmentation_name(augmentation_applied)
        aug_name = aug_name if aug_name else 'NoAug'

        # 증강된 이미지 저장
        augmented_image_file = f"{base_name}_aug_{aug_name}_{scale}.jpg"
        augmented_image_path = os.path.join(augmented_image_dir, augmented_image_file)
        cv2.imwrite(augmented_image_path, augmented_image)

        # 증강된 bounding box를 YOLO 형식으로 변환하여 저장
        augmented_label_file = f"{base_name}_aug_{aug_name}_{scale}.txt"
        augmented_label_path = os.path.join(augmented_label_dir, augmented_label_file)
        with open(augmented_label_path, 'w') as f:
            for bbox, class_id in zip(valid_bboxes, valid_class_ids):
                yolo_bbox = pascal_voc_to_yolo(bbox, img_width, img_height)  # Pascal VOC -> YOLO 변환
                yolo_bbox = [max(min(coord, 1.0), 0.0) for coord in yolo_bbox]  # 좌표 값 클리핑 (0~1 범위)
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        print(f"증강된 이미지 및 라벨 저장: {augmented_image_path}, {augmented_label_path}")

def augment_with_different_resize(image, bboxes, class_ids, img_width, img_height, base_name, different_limits, augmented_image_dir, augmented_label_dir):
    for new_size in different_limits:

        new_height = int(img_height * new_size)
        new_width = int(img_width * new_size)
        transform = A.ReplayCompose(
            [A.Resize(height=new_height, width=new_width)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'])
            )
        
        # 증강 적용
        augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
        augmentation_applied = augmented['replay']['transforms']  # Replay 정보 사용 (적용된 증강 방법 정보 저장)
        augmented_image = augmented['image']
        adjusted_bboxes = adjust_bbox_for_resize(bboxes, img_width, img_height, new_width, new_height)
        augmented_bboxes = adjusted_bboxes
        augmented_class_ids = augmented['class_ids']

        # 유효한 bounding box만 선택 (유효하지 않은 좌표는 필터링)
        valid_bboxes = []
        valid_class_ids = []
        for bbox, class_id in zip(augmented_bboxes, augmented_class_ids):
            x_min, y_min, x_max, y_max = bbox
            # 이미지 범위를 벗어나지 않는 유효한 박스만 유지
            if x_min >= 0 and y_min >= 0 and x_max <= img_width and y_max <= img_height and x_min < x_max and y_min < y_max:
                valid_bboxes.append(bbox)
                valid_class_ids.append(class_id)

        # 유효한 bounding box가 없으면 스킵
        if not valid_bboxes:
            continue

        # 증강 방법 이름 추출
        aug_name = get_augmentation_name(augmentation_applied)
        aug_name = aug_name if aug_name else 'NoAug'

        # 증강된 이미지 저장
        augmented_image_file = f"{base_name}_aug_{aug_name}_{new_size}.jpg"
        augmented_image_path = os.path.join(augmented_image_dir, augmented_image_file)
        cv2.imwrite(augmented_image_path, augmented_image)

        # 증강된 bounding box를 YOLO 형식으로 변환하여 저장
        augmented_label_file = f"{base_name}_aug_{aug_name}_{new_size}.txt"
        augmented_label_path = os.path.join(augmented_label_dir, augmented_label_file)
        with open(augmented_label_path, 'w') as f:
            for bbox, class_id in zip(valid_bboxes, valid_class_ids):
                yolo_bbox = pascal_voc_to_yolo(bbox, new_width, new_height)
                yolo_bbox = [max(min(coord, 1.0), 0.0) for coord in yolo_bbox]  # 좌표 값 클리핑 (0~1 범위)
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        print(f"증강된 이미지 및 라벨 저장: {augmented_image_path}, {augmented_label_path}")


# 이미지 크기를 변환하고 bounding box 좌표를 재조정하는 함수
def adjust_bbox_for_resize(original_bboxes, original_width, original_height, new_width, new_height):
    # 원본 좌표값을 새로운 이미지 크기에 맞게 변환
    adjusted_bboxes = []
    for bbox in original_bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_min = x_min * (new_width / original_width)
        y_min = y_min * (new_height / original_height)
        x_max = x_max * (new_width / original_width)
        y_max = y_max * (new_height / original_height)
        adjusted_bboxes.append([x_min, y_min, x_max, y_max])
    return adjusted_bboxes

# 폴더 경로 설정
image_dir = 'CarPlateProject/data/images/train'
label_dir = 'CarPlateProject/data/labels/train'
augmented_image_dir = 'CarPlateProject/data/images/train_augmented'
augmented_label_dir = 'CarPlateProject/data/labels/train_augmented'

# 증강 폴더가 없으면 생성
os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_label_dir, exist_ok=True)

# 이미지 파일 리스트 가져오기
image_files = os.listdir(image_dir)

# 선택한 증강 방법 리스트 (이 증강 방법은 항상 적용됨)
# 사용자가 선택한 증강 방법 리스트를 정의합니다. 이 증강 방법들이 이미지에 순차적으로 적용됩니다.
selected_augmentations = [
    'HorizontalFlip',
    'ShiftScaleRotate',
    'RandomBrightnessContrast',
    'MotionBlur',
    'GaussianBlur',
    'RandomRain',
    'RandomFog',
    'CLAHE',
    'RandomGamma',
    'HueSaturationValue',
    'Resize',
    'Affine'
]

# 선택한 증강 방법을 사용해 변환 객체 생성
all_selected_transforms = get_all_transforms(selected_augmentations)

# 이미지와 대응하는 라벨 파일 증강 및 저장
for image_file in image_files:
    if image_file.endswith('.jpg') or image_file.endswith('.png'):  # 이미지 파일 필터
        # 이미지 파일 경로
        image_path = os.path.join(image_dir, image_file)

        # 해당 이미지와 동일한 이름의 라벨 파일
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_file)

        # 이미지 불러오기
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape

        # 라벨 파일 읽기
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # YOLO 형식의 bounding box를 Pascal VOC 형식으로 변환
        bboxes = []
        class_ids = []
        for line in lines:
            elements = line.strip().split()
            class_id = elements[0]
            x_center, y_center, width, height = map(float, elements[1:])
            bbox = yolo_to_pascal_voc([x_center, y_center, width, height], img_width, img_height)
            bboxes.append(bbox)
            class_ids.append(int(class_id))

        # 원본 이미지와 라벨 저장
        base_name = os.path.splitext(image_file)[0]
        original_image_file = f"{base_name}_original.jpg"
        original_image_path = os.path.join(augmented_image_dir, original_image_file)
        cv2.imwrite(original_image_path, image)

        # 원본 bounding box를 YOLO 형식으로 변환하여 저장
        original_label_file = f"{base_name}_original.txt"
        original_label_path = os.path.join(augmented_label_dir, original_label_file)
        with open(original_label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_ids):
                yolo_bbox = pascal_voc_to_yolo(bbox, img_width, img_height)
                yolo_bbox = [max(min(coord, 1.0), 0.0) for coord in yolo_bbox]  # 좌표 값 클리핑
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        print(f"원본 이미지 및 라벨 저장: {original_image_path}, {original_label_path}")

        # 각 증강 방법을 개별적으로 적용
        for aug in all_selected_transforms:
            # Compose로 단일 증강 방법 적용
            if aug.__class__.__name__ == 'ShiftScaleRotate':
                scale_limits = [0.1, 0.2, 0.3, 0.4]
                # Scale 증강을 여러 값으로 시도하여 데이터 증가
                augment_with_different_scales(image, bboxes, class_ids, img_width, img_height, base_name, scale_limits, augmented_image_dir, augmented_label_dir)
                continue
            elif aug.__class__.__name__ == 'Resize':
                different_limits = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
                augment_with_different_resize(image, bboxes, class_ids, img_width, img_height, base_name, different_limits, augmented_image_dir, augmented_label_dir)
                continue
            else:
                transform = A.ReplayCompose([aug], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))

            # 증강 적용
            augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
            augmentation_applied = augmented['replay']['transforms']  # Replay 정보 사용
            augmented_image = augmented['image']

            # if aug.__class__.__name__ == 'Resize':
            #     # bounding box 크기 조정
            #     adjusted_bboxes = adjust_bbox_for_resize(bboxes, img_width, img_height, new_width, new_height)
            #     augmented_bboxes = adjusted_bboxes
            # else:
            augmented_bboxes = augmented['bboxes']
            augmented_class_ids = augmented['class_ids']

            # 유효한 bounding box만 선택 및 좌표 보정 (예: 90도 회전 처리)
            valid_bboxes = []
            valid_class_ids = []
            for bbox, class_id in zip(augmented_bboxes, augmented_class_ids):
                x_min, y_min, x_max, y_max = bbox
                if x_min >= 0 and y_min >= 0 and x_max <= img_width and y_max <= img_height and x_min < x_max and y_min < y_max:
                    valid_bboxes.append(bbox)
                    valid_class_ids.append(class_id)

            # 유효한 bounding box가 없으면 스킵
            if not valid_bboxes:
                continue

            # 증강 방법 이름 추출
            aug_name = get_augmentation_name(augmentation_applied)
            aug_name = aug_name if aug_name else 'NoAug'

            # 증강된 이미지 저장
            augmented_image_file = f"{base_name}_aug_{aug_name}.jpg"
            augmented_image_path = os.path.join(augmented_image_dir, augmented_image_file)
            cv2.imwrite(augmented_image_path, augmented_image)

            # 증강된 bounding box를 YOLO 형식으로 변환하여 저장
            augmented_label_file = f"{base_name}_aug_{aug_name}.txt"
            augmented_label_path = os.path.join(augmented_label_dir, augmented_label_file)
            with open(augmented_label_path, 'w') as f:
                for bbox, class_id in zip(valid_bboxes, valid_class_ids):
                    # if aug.__class__.__name__ == 'Resize':
                    #     yolo_bbox = pascal_voc_to_yolo(bbox, new_width, new_height)
                    # else:
                    yolo_bbox = pascal_voc_to_yolo(bbox, img_width, img_height)
                    yolo_bbox = [max(min(coord, 1.0), 0.0) for coord in yolo_bbox]  # 좌표 값 클리핑
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

            print(f"증강된 이미지 및 라벨 저장: {augmented_image_path}, {augmented_label_path}")
