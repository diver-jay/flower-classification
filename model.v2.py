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
from tensorflow.keras.regularizers import l2
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
MODEL_DIR = "flower_model_resnet50_anti_overfitting"
PLOT_DIR = "training_plots_resnet50_anti_overfitting"

# 디렉토리 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 설정 값
IMG_SIZE = 224  # ResNet50의 표준 입력 크기
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.00005  # 학습률 추가 감소
WEIGHT_DECAY = 1e-4  # L2 정규화 계수
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
        
        # 과적합 확인 (훈련 정확도와 검증 정확도의 차이가 크면 경고)
        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if train_acc - val_acc > 0.15:  # 15% 이상 차이나면 과적합 의심
            print(f"  [경고] 과적합 징후 감지: 훈련 정확도와 검증 정확도의 차이가 {(train_acc - val_acc)*100:.1f}%입니다.")

def create_model(num_classes):
    """
    과적합 방지를 위해 개선된 ResNet50 기반 전이 학습 모델 생성
    """
    # 사전 훈련된 ResNet50 로드 (가중치는 ImageNet)
    base_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # 배치 정규화 레이어의 파라미터 설정
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        layer.trainable = False  # 모든 기본 모델 레이어를 처음에는 동결
    
    # 모델 구조 정의 - 강화된 정규화가 적용된 분류 헤드
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 첫 번째 Dense 레이어에 L2 정규화 추가, 드롭아웃 비율 증가
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # 드롭아웃 비율 증가
    
    # 두 번째 Dense 레이어에 L2 정규화 추가, 드롭아웃 비율 증가
    x = Dense(512, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # 드롭아웃 비율 증가
    
    # 출력 레이어에도 L2 정규화 추가
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 기존 모델에서 더 적은 레이어만 동결 해제 (마지막 10개만 학습)
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    # 학습률 스케줄러 - 낮은 학습률 적용
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=500,  # 더 빨리 감소
        decay_rate=0.85,  # 더 많이 감소
        staircase=True
    )
    
    # 모델 컴파일 - weight decay 추가
    optimizer = Adam(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
    
    model.compile(
        optimizer=optimizer,
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
    # 기본 모델의 더 많은 레이어를 훈련 가능하게 설정 (여전히 처음 절반은 동결 유지)
    total_layers = len(base_model.layers)
    trainable_layers = total_layers // 4  # 전체의 1/4만 훈련
    
    for layer in base_model.layers:
        layer.trainable = False  # 모든 레이어 우선 동결
        
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True  # 마지막 1/4만 학습 가능하게
    
    # 더 낮은 학습률로 다시 컴파일
    optimizer = Adam(
        learning_rate=INITIAL_LEARNING_RATE / 5,
        weight_decay=WEIGHT_DECAY * 2  # 미세 조정 단계에서 weight decay 증가
    )
    
    model.compile(
        optimizer=optimizer,
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
    
    # 과적합 영역 강조 표시
    for i in range(len(history.history['accuracy'])):
        diff = history.history['accuracy'][i] - history.history['val_accuracy'][i]
        if diff > 0.15:  # 15% 이상 차이나면 과적합 영역으로 표시
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
    
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
    print("과적합 방지 ResNet50 기반 꽃 분류 모델 학습을 시작합니다...")
    
    # 클래스 목록 가져오기
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(class_names)
    print(f"발견된 꽃 클래스: {num_classes}개 - {class_names}")
    
    # 데이터 증강 설정 - 더 강한 증강 적용
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ResNet50 전용 전처리 함수
        validation_split=VALIDATION_SPLIT,
        # 증강 강화
        rotation_range=40,  # 더 큰 회전 각도
        width_shift_range=0.3,  # 더 큰 이동
        height_shift_range=0.3,  # 더 큰 이동
        shear_range=0.3,  # 더 큰 전단
        zoom_range=0.3,  # 더 큰 줌
        horizontal_flip=True,
        vertical_flip=True,  # 수직 뒤집기 추가
        brightness_range=[0.7, 1.3],  # 밝기 변화 추가
        fill_mode='reflect',  # 리플렉션 모드 사용
        # 추가 증강
        channel_shift_range=0.2  # 색상 변형 추가
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
    print("과적합 방지 ResNet50 기반 모델 생성 완료")
    
    # 모델 요약 정보 출력
    model.summary()
    
    # 배치당 손실 출력 콜백
    batch_loss_callback = BatchLossCallback(print_interval=5)
    
    # 콜백 설정 - 더 빠른 조기 중단
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
            patience=5,  # 인내심 감소로 과적합 방지
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # 인내심 감소
            min_lr=1e-7,
            verbose=1
        ),
        # 과적합 감지 및 학습률 감소 콜백 추가
        EarlyStopping(
            monitor='accuracy',
            patience=3,
            baseline=0.85,  # 훈련 정확도가 85%를 넘으면 주의
            verbose=1
        ),
        batch_loss_callback
    ]
    
    # 1단계: 동결 모델 학습 (더 적은 에폭)
    print("\n[1단계] 일부 레이어 학습 중...")
    history_initial = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS // 3,  # 에폭 수 감소하여 과적합 방지
        callbacks=callbacks,
        verbose=2  # 진행률 표시줄 없이 에폭당 한 줄씩 출력
    )
    
    plot_training_history(history_initial, "initial_training_history.png")
    
    # 2단계: 미세 조정 - 더 적은 층 고정 해제 후 재학습
    print("\n[2단계] 미세 조정 - 특징 추출기 일부 학습 중...")
    model = unfreeze_model(model, base_model)
    
    # 미세 조정 단계의 학습률은 더 낮게 설정
    history_fine_tuning = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS // 3,  # 에폭 수 감소
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
    
    # 과적합 정도 계산 - 최종 훈련 정확도와 검증 정확도의 차이
    final_train_acc = history_fine_tuning.history['accuracy'][-1]
    overfitting_gap = final_train_acc - test_acc
    print(f"과적합 정도: {overfitting_gap:.4f} ({overfitting_gap*100:.1f}%)")
    
    if overfitting_gap > 0.1:
        print("[경고] 여전히 과적합이 있습니다. 더 강한 정규화나 데이터 증강을 고려하세요.")
    else:
        print("[양호] 과적합이 효과적으로 제어되었습니다.")
    
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