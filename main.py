# main.py 파일에 이 코드를 모두 붙여넣고 저장하세요.

# =========================================
# 0. 필수 라이브러리 임포트 (가장 먼저 실행)
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("All necessary libraries imported!")
print(f"TensorFlow version: {tf.__version__}")
print("-" * 50)


# =========================================
# 1. CSV 파일 불러오기 (로컬 파일 시스템 사용)
# =========================================
# CSV 파일이 이 스크립트 파일과 같은 폴더에 있다고 가정합니다.
file_path = 'mlb_statcast_2021_2024_data.csv'

print(f"\n--- Loading data from {file_path} ---")
try:
    raw_data = pd.read_csv(file_path)
    print(f"raw_data loaded successfully! Total rows: {len(raw_data)}")
    print("First 5 rows of raw_data:")
    print(raw_data.head())
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the CSV file is in the same directory as this script.")
    print("Exiting program.")
    exit() # 파일이 없으면 프로그램 종료

print("-" * 50)


# =========================================
# 2. 데이터 전처리 (df DataFrame 생성)
# =========================================
print("\n--- Starting data preprocessing to create 'df' DataFrame ---")

# 필요한 컬럼 선택
selected_columns = [
    'events',
    'launch_speed',
    'launch_angle',
    'bb_type'
]
df = raw_data[selected_columns].copy()

print(f"DataFrame 'df' created with selected columns. Shape: {df.shape}")
print("First 5 rows of 'df':")
print(df.head())

# 결측치 처리
print("\nChecking for missing values before dropping:")
print(df.isnull().sum())
df.dropna(subset=['events', 'launch_speed', 'launch_angle', 'bb_type'], inplace=True)
print("\nMissing values after dropping NaNs:")
print(df.isnull().sum())
print(f"DataFrame shape after dropping NaNs: {df.shape}")

# 타겟 변수 'is_homerun' 생성
df['is_homerun'] = (df['events'] == 'home_run').astype(int)

# 타구가 아닌 이벤트 (삼진, 볼넷 등) 제거
non_batted_ball_events = [
    'walk', 'strikeout', 'hit_by_pitch', 'intent_walk', 'catcher_interf',
    'pitchout', 'sac_bunt_foul_tip', 'fielders_choice'
]
df = df[~df['events'].isin(non_batted_ball_events)].copy()

print("\n'is_homerun' target variable created and non-batted ball events removed.")
print("Homerun (1) vs Non-Homerun (0) counts:")
print(df['is_homerun'].value_counts())
print(f"Final DataFrame 'df' shape: {df.shape}")
print("First 5 rows of final 'df':")
print(df.head())
print("-" * 50)


# =========================================
# 3. 탐색적 데이터 분석 (EDA)
# =========================================
print("\n--- Starting Exploratory Data Analysis (EDA) ---")

# 3-1. 수치형 피처 (launch_speed, launch_angle) 분포 확인
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['launch_speed'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Launch Speed')
plt.xlabel('Launch Speed (mph)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.histplot(df['launch_angle'], kde=True, bins=50, color='lightcoral')
plt.title('Distribution of Launch Angle')
plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Count')
plt.tight_layout()
plt.show() # 로컬에서 그래프 띄우기

# 3-2. 홈런 vs 비홈런 타구의 발사각도/출구속도 분포 시각화
df_homerun = df[df['is_homerun'] == 1]
df_non_homerun = df[df['is_homerun'] == 0]

plt.figure(figsize=(10, 8))
plt.scatter(df_non_homerun['launch_speed'], df_non_homerun['launch_angle'],
            alpha=0.1, s=5, color='gray', label='Non-Homerun')
plt.scatter(df_homerun['launch_speed'], df_homerun['launch_angle'],
            alpha=0.5, s=15, color='red', label='Homerun')
plt.title('Launch Speed vs. Launch Angle for Homeruns vs. Non-Homeruns')
plt.xlabel('Launch Speed (mph)')
plt.ylabel('Launch Angle (degrees)')
plt.xlim(50, 120)
plt.ylim(-60, 90)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show() # 로컬에서 그래프 띄우기

# 3-3. bb_type (타구 유형) 분포 및 홈런 비율 확인
print("\nDistribution of Ball Type (bb_type):")
print(df['bb_type'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='bb_type', palette='viridis')
plt.title('Count of Ball Types')
plt.xlabel('Ball Type')
plt.ylabel('Count')
plt.show() # 로컬에서 그래프 띄우기

homerun_rates_by_bb_type = df.groupby('bb_type')['is_homerun'].mean().sort_values(ascending=False)
print("\nHomerun Rate by Ball Type:")
print(homerun_rates_by_bb_type)

plt.figure(figsize=(8, 6))
sns.barplot(x=homerun_rates_by_bb_type.index, y=homerun_rates_by_bb_type.values, palette='plasma')
plt.title('Homerun Rate by Ball Type')
plt.xlabel('Ball Type')
plt.ylabel('Homerun Rate')
plt.ylim(0, 1)
plt.show() # 로컬에서 그래프 띄우기

print("-" * 50)


# =========================================
# 4. 피처 엔지니어링 및 데이터 분할
# =========================================
print("\n--- Starting feature engineering and data splitting ---")

# 범주형 피처 (bb_type) 원-핫 인코딩
df_encoded = pd.get_dummies(df, columns=['bb_type'], prefix='bb_type')

# 피처 (X)와 타겟 (y) 분리
X = df_encoded[['launch_speed', 'launch_angle',
                'bb_type_ground_ball', 'bb_type_line_drive', 'bb_type_fly_ball', 'bb_type_popup']]
# Note: 'bb_type_popup'을 추가했습니다. 혹시 이전 코드에서 누락되었을 수 있습니다.
# get_dummies 결과에 따라 컬럼 목록을 확인하고 정확하게 맞춰주세요.
# df_encoded.columns를 출력하여 확인하는 것이 가장 좋습니다.
# print(df_encoded.columns)

y = df_encoded['is_homerun']

# 데이터 분할: 훈련, 검증, 테스트 세트
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# 피처 스케일링 (Standard Scaling for numerical features)
scaler = StandardScaler()
X_train[['launch_speed', 'launch_angle']] = scaler.fit_transform(
    X_train[['launch_speed', 'launch_angle']]
)
X_val[['launch_speed', 'launch_angle']] = scaler.transform(
    X_val[['launch_speed', 'launch_angle']]
)
X_test[['launch_speed', 'launch_angle']] = scaler.transform(
    X_test[['launch_speed', 'launch_angle']]
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("Features (launch_speed, launch_angle) have been scaled.")
print("X_train head after scaling:")
print(X_train.head())
print("-" * 50)


# =========================================
# 5. Keras Functional API 모델 구축 및 학습
# =========================================
print("\n--- Starting Keras Functional API Model Building and Training ---")

num_features = X_train.shape[1] # 입력 피처의 개수

# 입력 레이어 정의
input_layer = keras.Input(shape=(num_features,), name='input_features')

# 히든 레이어 구성
x = layers.Dense(64, activation='relu', name='hidden_layer_1')(input_layer)
x = layers.Dropout(0.3, name='dropout_1')(x)
x = layers.Dense(32, activation='relu', name='hidden_layer_2')(x)
x = layers.Dropout(0.3, name='dropout_2')(x)

# 출력 레이어 정의 (이진 분류)
output_layer = layers.Dense(1, activation='sigmoid', name='homerun_prediction')(x)

# 모델 생성
model = keras.Model(inputs=input_layer, outputs=output_layer, name='Homerun_Prediction_Model')

print("\n--- Model Summary ---")
model.summary()
print("\n")

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("Model compiled successfully!")

# 모델 학습
print("\n--- Starting model training ---")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
print("\n--- Model training complete! ---")

# 학습 과정 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show() # 로컬에서 그래프 띄우기

print("\n--- Keras Functional API model built, trained, and evaluation plots generated. ---")