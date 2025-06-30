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
from sklearn.preprocessing import StandardScaler, LabelEncoder # LabelEncoder는 이 주제에서 필수는 아님 (범주형 인코딩용)
from sklearn.metrics import mean_squared_error, mean_absolute_error # 회귀 평가 지표 추가
import os

print("All necessary libraries imported!")
print(f"TensorFlow version: {tf.__version__}")
print("-" * 50)

# =========================================
# 1. 기존 데이터 파일 불러오기 (로컬 파일 시스템 사용)
# =========================================
file_name = 'mlb_statcast_2021_2024_data.csv' # 실제 파일명과 확장자로 변경!

print(f"\n--- Loading data from {file_name} ---")
try:
    if file_name.endswith('.csv'):
        raw_data = pd.read_csv(file_name)
    elif file_name.endswith('.parquet'):
        raw_data = pd.read_parquet(file_name)
    else:
        print("Error: Unsupported file format. Please use .csv or .parquet files.")
        exit()

    print(f"raw_data loaded successfully! Total rows: {len(raw_data)}")
    print("First 5 rows of raw_data:")
    print(raw_data.head())
    # raw_data의 전체 컬럼명을 확인하는 코드 (디버깅용)
    # print("\nRaw data columns:", raw_data.columns.tolist())
except FileNotFoundError:
    print(f"Error: {file_name} not found. Please ensure the data file is in the same directory as this script.")
    print("Exiting program.")
    exit()
print("-" * 50)

# =========================================
# 2. 데이터 전처리 및 피처/타겟 분리
# =========================================
print("\n--- Starting data preprocessing ---")

# 타구 지점 예측에 필요한 컬럼 선택
# 'spin_rate'는 데이터에 없는 경우가 있으므로 제외하고, 필요한 컬럼만 선택
selected_cols = [
    'events', # 타구 결과 (필터링 용도)
    'launch_speed', 'launch_angle', # 핵심 피처
    'stand', 'p_throws', # 타자 유형 (좌/우), 투수 던지는 손 (좌/우)
    'pitch_type', # 투구 유형 (구질)
    'balls', 'strikes', 'outs_when_up', # 볼카운트, 아웃카운트
    'on_1b', 'on_2b', 'on_3b', # 주자 유무
    'hc_x', 'hc_y' # 타겟: 타구 착지 X, Y 좌표
    # 추가적으로 고려할 수 있는 피처: hit_distance_sc (예측할 타겟이 아니라면 피처로 사용 가능)
]
df = raw_data[selected_cols].copy()

# 1. 타구 이벤트만 필터링: Null이 아니며, 실제로 타구가 발생한 이벤트만.
# 'strikeout', 'walk', 'hit_by_pitch' 등 타구가 아닌 이벤트는 제외
batted_ball_events = [
    'field_out', 'single', 'double', 'triple', 'home_run',
    'force_out', 'grounded_into_double_play', 'sac_fly', 'sac_bunt', 'field_error'
]
df = df[df['events'].isin(batted_ball_events)].copy()

# 타겟 컬럼(hc_x, hc_y) 및 핵심 피처에 NaN이 없는 행만 사용
df.dropna(subset=['hc_x', 'hc_y', 'launch_speed', 'launch_angle'], inplace=True)

# 기타 피처의 NaN 처리 (예: pitch_type)
# pitch_type에 NaN이 있을 수 있으니 'Unknown'으로 채우거나 제거
df['pitch_type'].fillna('Unknown', inplace=True)

print(f"정제된 타구 데이터 수: {len(df)} 행")

# 2. 범주형 피처 원-핫 인코딩
# 'stand', 'p_throws', 'pitch_type'
df_encoded = pd.get_dummies(df, columns=['stand', 'p_throws', 'pitch_type'], prefix=['stand', 'p_throws', 'pitch_type'])

# 3. 주자 유무 컬럼을 int (0 또는 1)로 변환 (NaN은 0으로 채움)
df_encoded['on_1b'] = df_encoded['on_1b'].fillna(0).astype(int)
df_encoded['on_2b'] = df_encoded['on_2b'].fillna(0).astype(int)
df_encoded['on_3b'] = df_encoded['on_3b'].fillna(0).astype(int)

# 피처(X)와 타겟(y) 분리
# 타겟은 'hc_x', 'hc_y' 컬럼
X = df_encoded.drop(['events', 'hc_x', 'hc_y'], axis=1) # 'events'는 더 이상 필요 없음
y = df_encoded[['hc_x', 'hc_y']] # 타겟은 2개의 좌표

# 4. 수치형 피처 스케일링
# 원-핫 인코딩된 컬럼과 on_1b/2b/3b는 스케일링에서 제외
numerical_features = [
    'balls', 'strikes', 'outs_when_up',
    'launch_speed', 'launch_angle',
]
scaler_X = StandardScaler()
X[numerical_features] = scaler_X.fit_transform(X[numerical_features])

# 타겟(y)도 스케일링 (회귀 모델 성능 향상에 도움)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 5. 데이터 분할 (훈련, 검증, 테스트)
# 회귀 문제이므로 stratify는 사용하지 않습니다.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

print(f"\n훈련 데이터 크기: {X_train.shape}, {y_train.shape}")
print(f"검증 데이터 크기: {X_val.shape}, {y_val.shape}")
print(f"테스트 데이터 크기: {X_test.shape}, {y_test.shape}")
print("-" * 50)


# =========================================
# 3. Keras 딥러닝 모델 구축 및 학습 (Functional API 사용 권장)
# =========================================
print("\n--- Keras 딥러닝 모델 구축 ---")

# Functional API를 사용하여 다양한 피처들을 그룹별로 입력받는 모델 구성
# X.columns를 통해 각 그룹에 해당하는 컬럼들을 정의해야 합니다.
# 이 예시는 X의 컬럼 순서를 가정합니다. 실제 X.columns.tolist()를 출력하여 순서를 확인하세요.

# 물리적 피처 입력 (launch_speed, launch_angle)
input_physical = keras.Input(shape=(len(['launch_speed', 'launch_angle']),), name='physical_inputs')
# 상황 피처 입력 (balls, strikes, outs_when_up, on_1b, on_2b, on_3b)
input_context = keras.Input(shape=(len(['balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b']),), name='context_inputs')
# 범주형 피처 입력 (stand_*, p_throws_*, pitch_type_*) - 원핫 인코딩 후 남은 컬럼들
input_categorical = keras.Input(shape=(X.shape[1] - len(numerical_features) - len(['on_1b', 'on_2b', 'on_3b']),), name='categorical_inputs')
# on_1b, on_2b, on_3b는 numerical_features에 없지만, 원-핫 인코딩된 범주형 피처도 아니므로,
# input_context에 포함하거나 별도의 입력으로 처리할 수 있습니다.
# 여기서는 input_context에 포함했다고 가정하고, input_categorical의 shape 계산을 다시 합니다.
# 가장 정확한 방법은 X.columns.tolist()를 출력해서 input_categorical에 들어갈 컬럼 리스트를 직접 만드는 것입니다.

# 모든 피처가 X_train에 스케일링된 채로 들어오므로, Functional API를 사용하려면
# 각 입력 레이어에 어떤 컬럼들이 들어갈지 명확히 분리해야 합니다.

# 예시를 단순화하기 위해, 모든 피처를 하나의 입력으로 받는 Sequential 모델로 먼저 진행하는 것을 추천합니다.
# Functional API는 모델 학습이 성공한 후에 구조를 개선할 때 고려해볼 수 있습니다.
# (발표의 목적: 학습 과정을 명확히 보여주는 것)

num_features = X_train.shape[1] # 전체 피처 개수

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(num_features,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2) # 타겟이 hc_x, hc_y 두 개이므로 출력 노드 2개, 활성화 함수 없음 (회귀)
])

# 모델 컴파일 (회귀 모델이므로 loss는 MSE, MAE 측정)
model.compile(optimizer='adam',
              loss='mse', # Mean Squared Error
              metrics=['mae']) # Mean Absolute Error

print("\n모델 요약:")
model.summary()

# 모델 학습
print("\n모델 학습 시작...")
history = model.fit(
    X_train, y_train,
    epochs=100, # 충분한 에포크로 손실 감소 확인
    batch_size=128, # 배치 크기 (데이터가 많으므로)
    validation_data=(X_val, y_val),
    verbose=1
)
print("모델 학습 완료!")
print("-" * 50)


# =========================================
# 4. 학습 과정 시각화
# =========================================
print("\n--- 학습 과정 시각화 ---")

plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Loss over Epochs (Batted Ball Location)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# MAE 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE over Epochs (Batted Ball Location)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
print("-" * 50)


# =========================================
# 5. 모델 평가 및 전략 분석
# =========================================
print("\n--- 모델 최종 평가 ---")
# 테스트 세트에 대한 예측 수행 (스케일링된 값)
y_pred_scaled = model.predict(X_test)

# 예측값과 실제값을 원래 스케일로 역변환
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# 평가 지표 계산 (원래 스케일 기준)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae = mean_absolute_error(y_test_original, y_pred_original)

print(f"테스트 세트 RMSE (Root Mean Squared Error): {rmse:.2f} 피트")
print(f"테스트 세트 MAE (Mean Absolute Error): {mae:.2f} 피트")

# --- 타구 지점 시각화 (매우 중요, 발표 핵심!) ---
print("\n--- 모델 예측 타구 지점 시각화 ---")

plt.figure(figsize=(10, 10)) # 야구장 스케일에 맞게 조절
# 실제 타구 지점
plt.scatter(y_test_original[:, 0], y_test_original[:, 1], alpha=0.1, s=5, color='blue', label='Actual Locations')
# 예측 타구 지점 (오차 포함)
plt.scatter(y_pred_original[:, 0], y_pred_original[:, 1], alpha=0.1, s=5, color='red', label='Predicted Locations')

# 야구장 특정 지점 표시 (선택 사항)
# hc_x는 홈플레이트 중앙이 0, hc_y는 홈플레이트에서 타구 지점까지의 거리
# 대략적인 필드 경계 (Statcast 좌표는 피트 단위)
# 홈플레이트 중앙: (0, 0)
# 외야 펜스: 약 (0, 350~400) 피트, 좌우로 벌어짐
# 이 예시는 실제 야구장 스케일에 맞게 조정해야 합니다.
plt.xlim(-250, 250) # 홈플레이트 기준 좌우 250피트
plt.ylim(-50, 500) # 홈플레이트 뒤에서부터 외야 500피트
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.title('Actual vs Predicted Batted Ball Locations')
plt.xlabel('Horizontal Location (feet from home plate center)')
plt.ylabel('Vertical Location (feet from home plate)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_aspect('equal', adjustable='box') # x, y축 스케일을 동일하게
plt.show()

# --- 전략 분석 및 인사이트 (발표 핵심!) ---
print("\n--- 딥러닝 모델을 활용한 타구 전략 분석 및 인사이트 ---")

# 1. 평균적인 타구 지점 오차 설명 (MAE, RMSE 기반)
print(f"모델은 평균적으로 실제 타구 지점에서 약 {mae:.2f} 피트 정도의 오차로 예측합니다.")
print(f"RMSE는 {rmse:.2f} 피트이며, 이는 오차가 클수록 더 큰 페널티를 주는 지표입니다.")

# 2. 특정 조건에서의 타구 지점 예측 시뮬레이션
# (발사 속도/각도 조합, 투수/타자 유형 등 변경)

# X.columns.tolist()를 출력하여 정확한 컬럼 순서 확인 필요!
# 현재 X 컬럼 순서 (예시):
# ['balls', 'strikes', 'outs_when_up', 'on_1b', 'on_2b', 'on_3b',
#  'launch_speed', 'launch_angle',
#  'stand_L', 'stand_R', 'p_throws_L', 'p_throws_R',
#  'pitch_type_CH', 'pitch_type_CU', 'pitch_type_FF', ...] (나머지 원핫인코딩된 구질들)

# 전략 예시 1: 이상적인 '배럴 타구' 조건 (높은 발사 속도, 적정 발사 각도)
# (평균적인 볼카운트/주자/투수/타자 유형 사용)
# 임의의 값 대신, X_train의 평균값을 사용하여 '기본' 데이터를 만듭니다.
# 그리고 원하는 'launch_speed', 'launch_angle'만 변경합니다.

# X_train_df를 만들어서 컬럼명으로 접근하는 것이 오류를 줄이는 가장 좋은 방법입니다.
X_train_df = pd.DataFrame(X_train, columns=X.columns)

# 새로운 상황 데이터프레임 생성
new_pitch_data = pd.DataFrame(X_train_df.mean().values.reshape(1, -1), columns=X.columns)

# 105mph, 25도 타구 (이상적인 배럴 타구 조건)
new_pitch_data['launch_speed'] = scaler_X.transform(np.array([[105.0]])).item()
new_pitch_data['launch_angle'] = scaler_X.transform(np.array([[25.0]])).item()

# 예시: 우투수가 우타자에게 던진 포심 (원핫인코딩된 컬럼 조작)
# 이 부분은 실제 데이터의 원핫인코딩된 컬럼명에 맞춰야 합니다.
# X.columns.tolist()를 출력하여 'stand_R', 'p_throws_R', 'pitch_type_FF' 등이 어떤 위치에 있는지 확인
# 그리고 해당 컬럼을 1로, 나머지는 0으로 설정
# 예시: 'stand_L', 'stand_R' 중 'stand_R'만 1
for col in new_pitch_data.columns:
    if 'stand_' in col:
        new_pitch_data[col] = 0
    if 'p_throws_' in col:
        new_pitch_data[col] = 0
    if 'pitch_type_' in col:
        new_pitch_data[col] = 0

if 'stand_R' in new_pitch_data.columns: new_pitch_data['stand_R'] = 1
if 'p_throws_R' in new_pitch_data.columns: new_pitch_data['p_throws_R'] = 1
if 'pitch_type_FF' in new_pitch_data.columns: new_pitch_data['pitch_type_FF'] = 1


predicted_location_scaled = model.predict(new_pitch_data.values)[0]
predicted_location_original = scaler_y.inverse_transform(predicted_location_scaled.reshape(1, -1))[0]

print("\n--- 시뮬레이션 1: 105mph, 25도 타구 (우투수 -> 우타자, 포심) ---")
print(f"예상 타구 지점 (X, Y): ({predicted_location_original[0]:.2f}, {predicted_location_original[1]:.2f}) 피트")
# 이 지점을 야구장 어디에 해당하는지 설명합니다. (예: 좌측 담장 근처, 우중간 펜스 등)


# 전략 예시 2: 약한 땅볼 타구 조건 (낮은 발사 속도, 음수 발사 각도)
new_pitch_data_groundball = pd.DataFrame(X_train_df.mean().values.reshape(1, -1), columns=X.columns)
new_pitch_data_groundball['launch_speed'] = scaler_X.transform(np.array([[70.0]])).item()
new_pitch_data_groundball['launch_angle'] = scaler_X.transform(np.array([[-10.0]])).item()

# 동일한 투수/타자/구질 조건으로 가정
for col in new_pitch_data_groundball.columns:
    if 'stand_' in col: new_pitch_data_groundball[col] = 0
    if 'p_throws_' in col: new_pitch_data_groundball[col] = 0
    if 'pitch_type_' in col: new_pitch_data_groundball[col] = 0

if 'stand_R' in new_pitch_data_groundball.columns: new_pitch_data_groundball['stand_R'] = 1
if 'p_throws_R' in new_pitch_data_groundball.columns: new_pitch_data_groundball['p_throws_R'] = 1
if 'pitch_type_FF' in new_pitch_data_groundball.columns: new_pitch_data_groundball['pitch_type_FF'] = 1


predicted_location_gb_scaled = model.predict(new_pitch_data_groundball.values)[0]
predicted_location_gb_original = scaler_y.inverse_transform(predicted_location_gb_scaled.reshape(1, -1))[0]

print("\n--- 시뮬레이션 2: 70mph, -10도 타구 (약한 땅볼) ---")
print(f"예상 타구 지점 (X, Y): ({predicted_location_gb_original[0]:.2f}, {predicted_location_gb_original[1]:.2f}) 피트")
# 이 지점을 야구장 어디에 해당하는지 설명합니다. (예: 1루수 근처, 유격수 근처)

print("\n--- 분석 완료 ---")