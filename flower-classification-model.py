import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import time

# GPU 메모리 증가 방지
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU 설정 완료: {len(physical_devices)}개 발견")

# 경로 설정
DATA_DIR = "processed_flowers_dataset"
MODEL_DIR = "flower_model_resnet50_improved"
PLOT_DIR = "training_plots_resnet50_improved"

# 디렉토리 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 설정 값
IMG_SIZE = 224  # ResNet50의 표준 입력 크기
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_LEARNING_RATE = 0.0001  # 학습률 감소
VALIDATION_SPLIT = 0.2

# 각 배치마다 손실 및 정확도 출력을 위한 커스텀 콜백
class BatchLossCallback(Callback):
    def __init__(self, print_interval=5):
        super(BatchLossCallback, self).__init__()
        self.print_interval = print_interval
        self.batch_times = []
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n에폭 {epoch+1}/{self.params['epochs']} 시작")
    
    def on_batch_end(self, batch, logs=None):
        if batch % self.print_interval == 0:
            current_loss = logs.get('loss')
            current_accuracy = logs.get('accuracy')
            
            # 배치당 소요 시간 계산
            current_time = time.time()
            if hasattr(self, 'last_batch_time'):
                batch_time = current_time - self.last_batch_time
                self.batch_times.append(batch_time)
                avg_batch_time = sum(self.batch_times[-10:]) / min(len(self.batch_times), 10)
                time_str = f" - 배치 소요 시간: {batch_time:.4f}초 (평균: {avg_batch_time:.4f}초)"
            else:
                time_str = ""
                
            self.last_batch_time = current_time
            
            print(f"  배치 {batch}/{self.params['steps']}: 손실 = {current_loss:.4f}, 정확도 = {current_accuracy:.4f}{time_str}")
    
    def on_epoch_end(self, epoch, logs=None):
        # 에폭 소요 시간 계산
        epoch_time = time.time() - self.epoch_start_time
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')
        print(f"에폭 {epoch+1}/{self.params['epochs']} 완료 - 소요 시간: {epoch_time:.2f}초")
        print(f"  훈련 손실: {logs.get('loss'):.4f}, 훈련 정확도: {logs.get('accuracy'):.4f}")
        print(f"  검증 손실: {val_loss:.4f}, 검증 정확도: {val_accuracy:.4f}")

def create_model(num_classes):
    """
    개선된 ResNet50 기반 전이 학습 모델 생성
    """
    # 사전 훈련된 ResNet50 로드 (가중치는 ImageNet)
    base_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # 배치 정규화 레이어의 파라미터 설정
    # 이렇게 하면 훈련 중에 배치 정규화 통계가 업데이트되지 않음
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    # 모델 구조 정의 - 개선된 분류 헤드
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 배치 정규화 추가
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # 드롭아웃 비율 감소
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # 드롭아웃 비율 추가 감소
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 기존 모델 일부 레이어 동결 해제 (처음부터 일부 레이어 학습)
    # ResNet50 마지막 스테이지(stage 5)의 일부를 학습 가능하게 설정
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # 학습률 스케줄러 - 초기 학습에 낮은 학습률 사용
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 학습 가능한 레이어 수 계산 및 출력
    trainable_count = sum(layer.trainable for layer in model.layers)
    total_count = len(model.layers)
    print(f"총 레이어: {total_count}개, 학습 가능 레이어: {trainable_count}개")
    
    # 파라미터 수 계산
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = sum([np.prod(v.shape) for v in model.non_trainable_weights])
    print(f"학습 가능 파라미터: {trainable_params:,}")
    print(f"고정된 파라미터: {non_trainable_params:,}")
    
    return model, base_model

def unfreeze_model(model, base_model):
    """
    미세 조정을 위해 기본 모델의 더 많은 레이어 훈련 가능하게 설정
    """
    # 기본 모델의 더 많은 레이어를 훈련 가능하게 설정
    # ResNet50의 마지막 2개 스테이지(stage 4, 5)를 미세 조정
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # 더 낮은 학습률로 다시 컴파일
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 학습 가능한 레이어 수 계산 및 출력
    trainable_count = sum(layer.trainable for layer in model.layers)
    print(f"미세 조정 후 학습 가능 레이어: {trainable_count}개")
    
    # 파라미터 수 계산
    trainable_params = sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = sum([np.prod(v.shape) for v in model.non_trainable_weights])
    print(f"학습 가능 파라미터: {trainable_params:,}")
    print(f"고정된 파라미터: {non_trainable_params:,}")
    
    return model

def plot_training_history(history, filename="training_history.png"):
    """
    학습 과정 시각화
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 정확도 그래프
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 손실 그래프
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    """
    혼동 행렬 시각화
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    # 정규화된 혼동 행렬 (퍼센트 단위)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix (%)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    
    # 원래 값으로 된 혼동 행렬도 저장
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (counts)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"count_{filename}"))
    
    plt.close('all')

def train_flower_classifier():
    """
    꽃 분류 모델 학습 실행
    """
    print("개선된 ResNet50 기반 꽃 분류 모델 학습을 시작합니다...")
    
    # 클래스 목록 가져오기
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(class_names)
    print(f"발견된 꽃 클래스: {num_classes}개 - {class_names}")
    
    # 데이터 증강 설정 - ResNet50에 맞는 전처리
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ResNet50 전용 전처리 함수
        validation_split=VALIDATION_SPLIT,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # 검증 데이터는 데이터 증강 없이 전처리만 적용
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ResNet50 전용 전처리 함수
        validation_split=VALIDATION_SPLIT
    )
    
    # 학습용 데이터 생성기
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # 검증용 데이터 생성기
    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # 클래스 인덱스와 이름 매핑
    class_indices = train_generator.class_indices
    class_names_ordered = [None] * num_classes
    for class_name, idx in class_indices.items():
        class_names_ordered[idx] = class_name
    
    print("클래스 인덱스 매핑:", class_indices)
    
    # 모델 생성
    model, base_model = create_model(num_classes)
    print("개선된 ResNet50 기반 모델 생성 완료")
    
    # 모델 요약 정보 출력
    model.summary()
    
    # 배치당 손실 출력 콜백
    batch_loss_callback = BatchLossCallback(print_interval=5)
    
    # 콜백 설정
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # 인내심 증가
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # 인내심 증가
            min_lr=1e-7,
            verbose=1
        ),
        batch_loss_callback
    ]
    
    # 1단계: 부분 동결 모델 학습
    print("\n[1단계] 일부 레이어 학습 중...")
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS // 2,  # 절반의 에폭만 실행
        callbacks=callbacks,
        verbose=2  # 진행률 표시줄 없이 에폭당 한 줄씩 출력
    )
    
    plot_training_history(history_initial, "initial_training_history.png")
    
    # 2단계: 미세 조정 - 더 많은 층 고정 해제 후 재학습
    print("\n[2단계] 미세 조정 - 특징 추출기 더 많은 부분 학습 중...")
    model = unfreeze_model(model, base_model)
    
    # 미세 조정 단계의 학습률은 더 낮게 설정
    history_fine_tuning = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS // 2,  # 나머지 절반의 에폭 실행
        callbacks=callbacks,
        verbose=2  # 진행률 표시줄 없이 에폭당 한 줄씩 출력
    )
    
    plot_training_history(history_fine_tuning, "fine_tuning_history.png")
    
    # 전체 학습 과정 시각화
    full_history = {
        'accuracy': history_initial.history['accuracy'] + history_fine_tuning.history['accuracy'],
        'val_accuracy': history_initial.history['val_accuracy'] + history_fine_tuning.history['val_accuracy'],
        'loss': history_initial.history['loss'] + history_fine_tuning.history['loss'],
        'val_loss': history_initial.history['val_loss'] + history_fine_tuning.history['val_loss']
    }
    
    # 객체로 변환
    class History:
        def __init__(self, history):
            self.history = history
    
    plot_training_history(History(full_history), "full_training_history.png")
    
    # 모델 평가
    print("\n모델 평가 중...")
    validation_generator.reset()
    y_pred_prob = model.predict(validation_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = validation_generator.classes[:len(y_pred)]
    
    # 분류 보고서 출력
    report = classification_report(y_true, y_pred, target_names=class_names_ordered, output_dict=True)
    print("\n분류 보고서:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    report_df.to_csv(os.path.join(PLOT_DIR, 'classification_report.csv'))
    
    # 혼동 행렬 시각화
    plot_confusion_matrix(y_true, y_pred, class_names_ordered)
    
    # 모델 저장
    model.save(os.path.join(MODEL_DIR, 'flower_classifier_resnet50_improved.h5'))
    
    # TensorFlow Lite 모델 변환 (선택 사항)
    # 모바일 배포를 위해 TF Lite 모델로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # TFLite 모델 저장
    with open(os.path.join(MODEL_DIR, 'flower_classifier_model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    # 클래스 이름 저장
    with open(os.path.join(MODEL_DIR, 'class_names.txt'), 'w') as f:
        for class_name in class_names_ordered:
            f.write(f"{class_name}\n")
    
    # 최종 테스트 세트 정확도 계산
    test_loss, test_acc = model.evaluate(validation_generator, verbose=1)
    print(f"\n최종 테스트 정확도: {test_acc:.4f}")
    print(f"최종 테스트 손실: {test_loss:.4f}")
    
    print(f"\n모델 학습 완료! 모델이 {os.path.abspath(MODEL_DIR)} 폴더에 저장되었습니다.")
    print(f"클래스 목록: {class_names_ordered}")
    
    # 클래스별 정확도 계산
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names_ordered):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:
            class_correct = np.sum(y_pred[class_indices] == i)
            class_acc = class_correct / len(class_indices)
            per_class_accuracy[class_name] = class_acc
            print(f"클래스 '{class_name}' 정확도: {class_acc:.4f} ({class_correct}/{len(class_indices)})")
    
    return model, class_names_ordered

# 메인 실행
if __name__ == "__main__":
    train_flower_classifier()