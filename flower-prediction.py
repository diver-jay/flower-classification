import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import random
from PIL import Image
import cv2
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 경로 설정
DATA_DIR = "processed_flowers_dataset"
MODEL_DIR = "flower_model_resnet50"
RESULTS_DIR = "evaluation_results"
IMG_SIZE = 224

# 결과 디렉토리 생성
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_class_names():
    """
    저장된 클래스 이름 로드
    """
    class_names = []
    with open(os.path.join(MODEL_DIR, 'class_names.txt'), 'r') as f:
        for line in f:
            class_names.append(line.strip())
    return class_names

def preprocess_image(img_path):
    """
    이미지 전처리
    """
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 정규화
    return img_array

def get_random_images(data_dir, num_images=10, seed=None):
    """
    각 클래스에서 무작위 이미지 선택
    """
    if seed is not None:
        random.seed(seed)
        
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    selected_images = []
    
    # 각 클래스에서 균등하게 이미지 선택
    images_per_class = max(1, num_images // len(class_dirs))
    
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            continue
            
        # 각 클래스에서 이미지 무작위 선택
        selected = random.sample(image_files, min(images_per_class, len(image_files)))
        
        for img_file in selected:
            selected_images.append({
                'path': os.path.join(class_path, img_file),
                'true_class': class_dir
            })
        
    # 원하는 총 이미지 수에 맞게 무작위로 선택
    if len(selected_images) > num_images:
        selected_images = random.sample(selected_images, num_images)
    
    return selected_images

def predict_and_visualize(model, class_names, test_images, filename="prediction_results.png"):
    """
    모델 예측 결과 시각화
    """
    plt.figure(figsize=(15, 20))
    
    correct_count = 0
    total_count = len(test_images)
    confidence_scores = []
    
    for i, img_data in enumerate(test_images):
        # 이미지 로드 및 전처리
        img_path = img_data['path']
        true_class = img_data['true_class']
        true_class_idx = class_names.index(true_class) if true_class in class_names else -1
        
        img_array = preprocess_image(img_path)
        
        # 예측
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100
        
        # 신뢰도 저장
        confidence_scores.append(confidence)
        
        # 정확도 계산
        is_correct = (predicted_class == true_class)
        if is_correct:
            correct_count += 1
        
        # 원본 이미지 로드 (시각화용)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 시각화
        plt.subplot(5, 2, i+1)
        plt.imshow(img)
        
        # 예측이 맞았는지 여부에 따라 제목 색상 변경
        title_color = 'green' if is_correct else 'red'
        
        plt.title(f"실제: {true_class}\n예측: {predicted_class} ({confidence:.1f}%)", 
                 color=title_color, fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    
    # 정확도 및 평균 신뢰도 출력
    accuracy = correct_count / total_count * 100
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    
    print(f"\n테스트 이미지 예측 결과:")
    print(f"  - 정확도: {accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"  - 평균 신뢰도: {avg_confidence:.2f}%")
    
    return accuracy, avg_confidence

def create_confidence_distribution(model, class_names, data_dir, num_samples=100, filename="confidence_distribution.png"):
    """
    각 클래스별 예측 신뢰도 분포 시각화
    """
    # 각 클래스별 테스트 이미지 선택
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    confidence_by_class = {class_name: [] for class_name in class_names}
    
    for class_dir in class_dirs:
        if class_dir not in class_names:
            continue
            
        class_path = os.path.join(data_dir, class_dir)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 각 클래스에서 최대 num_samples/len(class_dirs) 개의 이미지 선택
        samples_per_class = min(len(image_files), num_samples // len(class_dirs))
        selected_files = random.sample(image_files, samples_per_class)
        
        for img_file in selected_files:
            img_path = os.path.join(class_path, img_file)
            img_array = preprocess_image(img_path)
            
            # 예측
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            # 예측이 실제 클래스와 일치할 경우에만 저장
            if predicted_class == class_dir:
                confidence_by_class[class_dir].append(confidence)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    for class_name, confidences in confidence_by_class.items():
        if confidences:  # 빈 목록이 아닌 경우만 처리
            sns.kdeplot(confidences, label=f"{class_name} (n={len(confidences)})")
    
    plt.title('신뢰도 분포 (정확한 예측만)', fontsize=14)
    plt.xlabel('신뢰도 (%)', fontsize=12)
    plt.ylabel('밀도', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    
    return confidence_by_class

def visualize_activation_maps(model, img_path, class_names, filename="activation_maps.png"):
    """
    모델의 활성화 맵 시각화 (Class Activation Mapping)
    """
    # 이미지 로드 및 전처리
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    # 원본 이미지
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # ResNet50의 마지막 컨볼루션 레이어 이름 (ResNet50의 경우 'conv5_block3_out')
    last_conv_layer_name = "conv5_block3_out"
    
    # Grad-CAM 구현
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(x)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]
    
    # 출력 특성 맵에 대한 클래스 활성화의 기울기 계산
    grads = tape.gradient(loss, conv_output)
    
    # 중요도 가중치 계산을 위해 채널 차원에서 평균
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 특성 맵과 가중치의 가중 조합
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # 히트맵 리사이즈 및 시각화
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 원본 이미지에 히트맵 오버레이
    superimposed_img = heatmap * 0.4 + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')