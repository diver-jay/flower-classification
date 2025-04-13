import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# 설정 값
INPUT_DIR = "korean_flowers_dataset"  # 원본 이미지 폴더
OUTPUT_DIR = "processed_flowers_dataset"  # 처리된 이미지 저장 폴더
TARGET_SIZE = (224, 224)  # 모든 이미지를 동일한 크기로 조정 (224x224는 많은 CNN 모델의 표준 입력 크기)
AUGMENT_FACTOR = 5  # 각 이미지당 생성할 증강 이미지 수 (증가)

# 출력 디렉토리 생성 또는 초기화
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)  # 기존 폴더 삭제
os.makedirs(OUTPUT_DIR)  # 새 폴더 생성

# 이미지 증강 함수
def augment_image(image, seed=None):
    """
    이미지에 다양한 변환을 적용하여 증강합니다.
    개선된 버전: 색조/채도 변경, 노이즈, 블러, 투시 변환, 확장된 회전 범위 등 추가
    """
    if seed is not None:
        random.seed(seed)
    
    # PIL 이미지로 변환
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    augmented_images = []
    
    # 원본 이미지 추가
    augmented_images.append(image)
    
    # 증강 1: 회전 (범위 확장: -30도 ~ 30도)
    angle = random.uniform(-30, 30)
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False)
    augmented_images.append(rotated)
    
    # 증강 2: 밝기 조정
    brightness_factor = random.uniform(0.7, 1.3)  # 범위 확장
    brightness_img = ImageEnhance.Brightness(image).enhance(brightness_factor)
    augmented_images.append(brightness_img)
    
    # 증강 3: 대비 조정
    contrast_factor = random.uniform(0.7, 1.3)  # 범위 확장
    contrast_img = ImageEnhance.Contrast(image).enhance(contrast_factor)
    augmented_images.append(contrast_img)
    
    # 증강 4: 좌우 반전
    flip_img = ImageOps.mirror(image)
    augmented_images.append(flip_img)
    
    # 증강 5: 자르기 및 크기 조정 (Random crop)
    width, height = image.size
    crop_size = min(width, height) * random.uniform(0.7, 0.9)  # 더 다양한 크기로 자르기
    left = random.uniform(0, width - crop_size)
    top = random.uniform(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    
    cropped = image.crop((left, top, right, bottom))
    resized_crop = cropped.resize(image.size, Image.BICUBIC)
    augmented_images.append(resized_crop)
    
    # 새 증강 6: 색조 변경 (Hue)
    if random.random() > 0.5:  # 50% 확률로 적용
        # 더 작은 범위의 색조 변경 (-5~5)으로 제한
        h_shift = random.randint(-5, 5)
    
        # PIL에서 HSV로 변환
        hue_shifted = image.convert('HSV')
        h, s, v = hue_shifted.split()
    
        # NumPy 배열로 변환하지 않고 ImageMath 사용
        from PIL import ImageMath
    
        # ImageMath를 사용하여 더 안전하게 색조 조정
        h = ImageMath.eval('(a + b) % 256', a=h, b=h_shift).convert('L')
    
        # 다시 합치기
        hue_shifted = Image.merge('HSV', (h, s, v)).convert('RGB')
        augmented_images.append(hue_shifted)

    # 새 증강 7: 채도 변경 (Saturation)
    saturation_factor = random.uniform(0.8, 1.3)
    saturation_img = ImageEnhance.Color(image).enhance(saturation_factor)
    augmented_images.append(saturation_img)
    
    # 새 증강 8: 가우시안 블러
    if random.random() > 0.5:  # 50% 확률로 적용
        blur_radius = random.uniform(0.5, 1.5)  # 약한 블러 적용
        blurred_img = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        augmented_images.append(blurred_img)
    
    # 새 증강 9: 노이즈 추가 (PIL로 구현)
    def add_noise(img):
        img_array = np.array(img)
        # 가우시안 노이즈 추가 (약한 정도)
        noise = np.random.normal(0, 15, img_array.shape).astype(np.uint8)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    if random.random() > 0.5:  # 50% 확률로 적용
        noisy_img = add_noise(image)
        augmented_images.append(noisy_img)
    
    # 새 증강 10: 투시 변환 (Perspective Transform)
    def perspective_transform(img):
        width, height = img.size
        
        # 원본 이미지의 네 모서리 좌표
        src_points = np.float32([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ])
        
        # 목표 좌표 (약간의 왜곡 추가)
        # 10% 이내의 왜곡만 적용하여 현실성 유지
        distortion = 0.1
        dst_points = np.float32([
            [random.uniform(0, width * distortion), random.uniform(0, height * distortion)],
            [random.uniform(width * (1-distortion), width), random.uniform(0, height * distortion)],
            [random.uniform(0, width * distortion), random.uniform(height * (1-distortion), height)],
            [random.uniform(width * (1-distortion), width), random.uniform(height * (1-distortion), height)]
        ])
        
        # OpenCV 형식으로 변환하여 투시 변환 수행
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(img_cv, M, (width, height))
        warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        
        return warped_pil
    
    if random.random() > 0.7:  # 30% 확률로 적용 (왜곡이 심할 수 있으므로)
        perspective_img = perspective_transform(image)
        augmented_images.append(perspective_img)
    
    # 랜덤하게 AUGMENT_FACTOR 개수만큼 선택
    selected = random.sample(augmented_images, min(AUGMENT_FACTOR, len(augmented_images)))
    
    # OpenCV 형식으로 변환하여 반환
    result = []
    for img in selected:
        result.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    
    return result

def process_dataset():
    """
    전체 데이터셋을 처리하고 증강합니다.
    """
    flower_types = []
    
    # 꽃 폴더 스캔
    for root, dirs, files in os.walk(INPUT_DIR):
        for dir_name in dirs:
            if os.path.isdir(os.path.join(INPUT_DIR, dir_name)):
                flower_types.append(dir_name)
    
    print(f"발견된 꽃 종류: {len(flower_types)}")
    
    # 클래스별 이미지 수 확인 (데이터 불균형 확인용)
    class_counts = {}
    for flower_type in flower_types:
        flower_dir = os.path.join(INPUT_DIR, flower_type)
        if os.path.isdir(flower_dir):
            image_files = [f for f in os.listdir(flower_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[flower_type] = len(image_files)
    
    # 데이터 불균형 확인 및 출력
    print("\n원본 데이터셋 클래스 분포:")
    for flower_type, count in class_counts.items():
        print(f"{flower_type}: {count}장")
    
    # 이미지 증강 처리
    for flower_type in flower_types:
        # 현재 꽃 종류 폴더
        flower_dir = os.path.join(INPUT_DIR, flower_type)
        if not os.path.isdir(flower_dir):
            continue
            
        # 출력 폴더 생성
        output_flower_dir = os.path.join(OUTPUT_DIR, flower_type)
        os.makedirs(output_flower_dir, exist_ok=True)
        
        # 이미지 파일 목록 가져오기
        image_files = [f for f in os.listdir(flower_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n처리 중: {flower_type} - 발견된 이미지: {len(image_files)}개")
        
        # 데이터 불균형이 심한 경우, 적은 클래스에 더 많은 증강 적용
        avg_count = sum(class_counts.values()) / len(class_counts)
        global AUGMENT_FACTOR
        class_augment_factor = AUGMENT_FACTOR
        if class_counts[flower_type] < avg_count * 0.7:  # 평균보다 30% 이상 적은 경우
            class_augment_factor = min(AUGMENT_FACTOR + 2, 8)  # 최대 2개 더 추가 (최대 8개)
            print(f"  - 데이터 불균형 감지: 증강 계수 증가 ({AUGMENT_FACTOR} -> {class_augment_factor})")
        
        # 각 이미지 처리
        for idx, img_file in enumerate(tqdm(image_files, desc=f"처리 중: {flower_type}")):
            img_path = os.path.join(flower_dir, img_file)
            
            try:
                # 이미지 로드
                img = cv2.imread(img_path)
                if img is None:
                    print(f"경고: 이미지를 로드할 수 없음 - {img_path}")
                    continue
                
                # 이미지 크기 조정 (비율 유지)
                h, w = img.shape[:2]
                if h > w:
                    new_h, new_w = int(TARGET_SIZE[1] * h / w), TARGET_SIZE[0]
                else:
                    new_h, new_w = TARGET_SIZE[1], int(TARGET_SIZE[0] * w / h)
                
                # 크기 조정
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 중앙 크롭
                h, w = resized.shape[:2]
                start_h = max(0, h // 2 - TARGET_SIZE[1] // 2)
                start_w = max(0, w // 2 - TARGET_SIZE[0] // 2)
                cropped = resized[start_h:start_h + TARGET_SIZE[1], start_w:start_w + TARGET_SIZE[0]]
                
                # 필요한 경우 패딩 추가
                h, w = cropped.shape[:2]
                if h < TARGET_SIZE[1] or w < TARGET_SIZE[0]:
                    top = (TARGET_SIZE[1] - h) // 2
                    bottom = TARGET_SIZE[1] - h - top
                    left = (TARGET_SIZE[0] - w) // 2
                    right = TARGET_SIZE[0] - w - left
                    cropped = cv2.copyMakeBorder(
                        cropped, top, bottom, left, right, 
                        cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )
                
                # 증강된 이미지 생성 (클래스별 조정된 증강 계수 사용)
                temp_augment_factor = AUGMENT_FACTOR
                AUGMENT_FACTOR = class_augment_factor
                augmented_images = augment_image(cropped, seed=idx)
                AUGMENT_FACTOR = temp_augment_factor  # 원래 값으로 복원
                
                # 이미지 저장
                for aug_idx, aug_img in enumerate(augmented_images):
                    file_name = f"{idx:04d}_aug{aug_idx}.jpg"
                    output_path = os.path.join(output_flower_dir, file_name)
                    cv2.imwrite(output_path, aug_img)
            
            except Exception as e:
                print(f"오류: {img_path} 처리 중 - {str(e)}")

    # 전처리 통계 출력
    print("\n전처리 및 증강 완료!")
    print(f"출력 폴더: {os.path.abspath(OUTPUT_DIR)}")
    
    # 각 클래스별 이미지 수 계산 (증강 후)
    total_images = 0
    print("\n증강 후 데이터셋 클래스 분포:")
    for flower_type in flower_types:
        output_flower_dir = os.path.join(OUTPUT_DIR, flower_type)
        if os.path.exists(output_flower_dir):
            class_images = len([f for f in os.listdir(output_flower_dir) if f.endswith('.jpg')])
            print(f"{flower_type}: {class_images}장 (원본: {class_counts[flower_type]}장, 증강비율: {class_images/class_counts[flower_type]:.1f}배)")
            total_images += class_images
    
    print(f"총 이미지 수: {total_images}장")

if __name__ == "__main__":
    process_dataset()