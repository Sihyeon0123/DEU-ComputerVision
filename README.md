# 컴퓨터비전 실습 코드 README

이 문서는 컴퓨터비전 과목에서 배운 실습 코드 내용을 정리한 것입니다. 각 실습의 목표, 주요 개념, 내용 요약 및 예제 코드를 포함하고 있습니다.

## 목차

- [SLP (Single Layer Perceptron)](#slp-single-layer-perceptron)
  - [AND, OR, XOR](#and-or-xor)
- [MLP (Multi-Layer Perceptron)](#mlp-multi-layer-perceptron)
  - [XOR](#xor)
  - [MNIST 데이터셋을 이용한 손글씨 인식](#mnist-데이터셋을-이용한-손글씨-인식)
- [CNN (Convolutional Neural Network)](#cnn-convolutional-neural-network)
  - [손글씨 인식](#손글씨-인식)
  - [패션 인식](#패션-인식)
- [Detectron2 기반 실습](#detectron2-기반-실습)
  - [차량 번호판 인식](#차량-번호판-인식)
  - [사람 스켈레톤 검출 및 프레임간 각도 변화를 통한 쓰러짐 검출](#사람-스켈레톤-검출-및-프레임간-각도-변화를-통한-쓰러짐-검출)

---

## SLP (Single Layer Perceptron)

### AND, OR, XOR

#### 주제
- 단일 계층 퍼셉트론(SLP)을 이용하여 논리 연산 AND, OR, XOR 구현.

#### 내용 요약
- SLP를 사용하여 AND, OR, XOR 논리 연산을 학습시킴.
- XOR 연산은 선형 분리가 불가능하여 SLP로는 해결할 수 없음.

#### 중요 개념
- 퍼셉트론: 인공 뉴런 모델로, 가중치와 편향을 통해 입력 값을 처리.
- 활성화 함수: 입력 값의 합이 특정 임계값을 넘으면 출력 값을 1로 설정, 그렇지 않으면 0으로 설정.

#### 예제 코드
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim = 2, activation='sigmoid'))
sgd = tf.keras.optimizers.SGD(learning_rate=0.05)
model.compile(loss='mean_squared_error',optimizer=sgd)
```

---

## MLP (Multi-Layer Perceptron)

### XOR

#### 주제
- 다층 퍼셉트론(MLP)을 이용하여 XOR 문제 해결.

#### 내용 요약
- MLP를 사용하여 XOR 문제를 해결함.
- 은닉층을 추가하여 비선형 분리를 가능하게 함.

#### 중요 개념
- 다층 퍼셉트론: 여러 개의 은닉층을 갖는 신경망 모델.
- 비선형 활성화 함수: ReLU, Sigmoid 등을 사용하여 비선형성을 도입.

#### 예제 코드
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, input_dim = 2, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
sgd = tf.keras.optimizers.SGD(learning_rate=0.05)
model.compile(loss='mean_squared_error',optimizer=sgd)
```

### MNIST 데이터셋을 이용한 손글씨 인식
#### 주제
- MLP를 이용하여 MNIST 데이터셋의 손글씨 숫자 인식.

#### 내용 요약
- MNIST 데이터셋을 사용하여 손글씨 숫자 인식을 수행.
- 28x28 픽셀의 이미지를 입력으로 받아 0-9까지의 숫자를 분류.

#### 중요 개념
- MNIST 데이터셋: 60,000개의 학습용 데이터와 10,000개의 테스트용 데이터로 구성된 손글씨 숫자 이미지 데이터셋.
- 분류 문제: 주어진 입력 이미지를 10개의 클래스 중 하나로 분류.

#### 예제 코드
```python
# 은닉 계층: RELU, 출력함수: sotfmax 계층 생성
# 128개의 노드를 가지는 은닉 RELU 1계층과 softmax 활성함수를 가지는 10개의 노드 출력층
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten: 2차원 -> 1차원으로 변형
  tf.keras.layers.Dense(128, activation='relu'),# 계층의 개수 변경, relu대신 다른 함수 사용
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 5번 학습 수행
model.fit(x_train, y_train, epochs=5, verbose=1)
```

---

## CNN (Convolutional Neural Network)
### 손글씨 인식
#### 주제
- CNN을 이용하여 MNIST 데이터셋의 손글씨 숫자 인식.

#### 내용 요약
- CNN을 사용하여 손글씨 숫자 인식을 수행.
- 컨볼루션 레이어와 풀링 레이어를 활용하여 이미지 특징 추출.

#### 중요 개념
- 컨볼루션 레이어: 이미지의 공간적 계층 구조를 학습.
- 풀링 레이어: 공간적 차원을 축소하여 중요한 특징만 추출.

#### 예제 코드
```python
# 모델 정의
model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
                          activation='relu', input_shape=(28, 28,1)),
   tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
   tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), padding='same', 
                          activation='relu'),
   tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1)
```

### 패션 인식
#### 주제
- CNN을 이용하여 패션 아이템 이미지 분류.

#### 내용 요약
- Fashion MNIST 데이터셋을 사용하여 패션 아이템 이미지를 분류.
- 다양한 패션 아이템을 10개의 클래스로 분류.

#### 중요 개념
-Fashion MNIST 데이터셋: 의류, 신발, 가방 등의 이미지 데이터셋.
- 이미지 분류: 주어진 이미지를 10개의 클래스 중 하나로 분류.
#### 예제 코드
```python
# 모델 정의
# 필터 32, 필터크기 3x3, 
model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same',
                          activation='relu', input_shape=(28, 28,1)),
   tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
   tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=(1,1), padding='same', 
                          activation='relu'),
   tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(10, activation='softmax')
])
# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 모델 학습
model.fit(x_train, y_train, epochs=5, verbose=1)
```

---

## Detectron2 기반 실습
### 차량 번호판 인식
#### 주제
- Detectron2를 이용하여 차량 번호판 인식.

#### 내용 요약
- Detectron2를 사용하여 차량 번호판을 인식하고 해당 영역을 추출.
- Pre-trained 모델을 활용하여 번호판 인식 성능을 향상.

##### 중요 개념
- Detectron2: Facebook AI Research에서 개발한 객체 탐지 라이브러리.
- 객체 탐지: 이미지에서 객체의 위치와 종류를 인식.

### 사람 스켈레톤 검출 및 프레임간 각도 변화를 통한 쓰러짐 검출
#### 주제
- Detectron2를 이용하여 사람의 스켈레톤 검출 및 프레임 간 각도 변화를 통한 쓰러짐 검출.

#### 내용 요약
- Detectron2의 Keypoint R-CNN을 사용하여 사람의 스켈레톤을 검출.
- 프레임 간 각도 변화를 분석하여 쓰러짐을 검출.

#### 중요 개념
- Keypoint R-CNN: 사람의 관절 위치를 검출하는 모델.
- 쓰러짐 검출: 관절의 각도 변화를 통해 쓰러짐을 인식.
