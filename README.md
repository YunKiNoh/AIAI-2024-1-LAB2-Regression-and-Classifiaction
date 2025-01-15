# AIAI(AI활용프로그래밍) LAB2: Regression and Classifiaction
본 레포지토리는 2024년 1학기 AI활용프로그래밍 강의의 두 번째 과제인 'Assignment2: Regression and Classification'의 내용을 정리 및 소개하기 위하여 작성하였습니다. 본 과제는 총 두 Part로 나뉘어져 있는데, 2-1은 Regression에 대한 것이고 2-2와 2-3은 Classification에 대한 내용입니다. '2-1:Regression'의 경우 전복의 여러가지 데이터를 통해 전복 나이를 예측하였으며, '2-2,2-3: Classification'의 경우 각 사람에 대한 데이터를 통해 비만의 정도를 예측하였습니다. 본 과제들은 모두 오픈 데이터셋을 활용하였습니다.

## Part1: Predict Abalone's Rings information by Using Regression Model
### 1. Dataset Information
Part1에서는 전복의 8가지 데이터를 활용하여 나이(Rings)를 예측하였습니다. 데이터 셋은 'UC Ivrine Machine Learning Repository'의 'Avalone' 오픈 데이터셋으로써 데이터에 대한 정보는 다음과 같습니다. [dataset link: https://archive.ics.uci.edu/dataset/1/abalone]

<div align="center">

| Variable Name   | Role     | Type         | Description                          | Units  | Missing Values |
| :-------------- | :------- | :----------- | :----------------------------------- | :----- | :------------- |
| Sex             | Feature  | Categorical  | M, F, and I (infant)                | -      | no             |
| Length          | Feature  | Continuous   | Longest shell measurement           | mm     | no             |
| Diameter        | Feature  | Continuous   | Perpendicular to length             | mm     | no             |
| Height          | Feature  | Continuous   | With meat in shell                  | mm     | no             |
| Whole_weight    | Feature  | Continuous   | Whole abalone                       | grams  | no             |
| Shucked_weight  | Feature  | Continuous   | Weight of meat                      | grams  | no             |
| Viscera_weight  | Feature  | Continuous   | Gut weight (after bleeding)         | grams  | no             |
| Shell_weight    | Feature  | Continuous   | After being dried                   | grams  | no             |
| Rings           | Target   | Integer      | +1.5 gives the age in years         | -      | no             |

<p style="margin-top: 10px;">Table 1. Abalone Dataset Description</p>

</div>

feature data는 총 1개의 Categorical data(성별)와 총 7개의 Numerical data(길이, 지름, 높이, 전체 무게, 살만 포함하는 무게, 내장 무게, 건조된 후의 껍질 무게)로 총 8가지 데이터 셋으로 나뉩니다.


<div align="center">
  
![image](https://github.com/user-attachments/assets/633f0893-3f30-41be-a19b-221e25bf3155)
<p style="margin-top: 10px;">Figure 1. Categorical Feature Data</p>
</div>


<div align="center">
  
![image](https://github.com/user-attachments/assets/fac4a2ce-0c81-4dce-8dbd-2658f14bf296)
<p style="margin-top: 10px;">Figure 2. Numerical Feature Data</p>
</div>

데이터 시각화를 통하여 전복의 성별 및 다른 수치 데이터의 양상을 확인할 수 있습니다. 눈여겨 볼만한 점은 I(Infants) 성별에 포함되어 있다는 점인데, 해당 군체의 경우 암컷 및 수컷 갯수와 큰 차이가 없다는 점에서 전체 전복의 대략 30% 가량은 비교적 나이가 적을 수 있음을 암시하고 있음을 알 수 있습니다.

이번 과제에서는 전복의 나이 값(Rings)을 맞춰야 하는데, 이때 이처럼 목표가 되는 데이터를 예측 목표 값인 target data이라고 부릅니다. 

<div align="center">
  
![image](https://github.com/user-attachments/assets/822a6f41-885d-461a-93b8-f568c697b672)

<p style="margin-top: 10px;">Figure 3. Target Data/p>
</div>

다른 feature data를 통해 전복의 나이를 예측하기 위해서는 regression(회귀)를 통하여 오차가 최소가 되는 모델을 학습시켜야 합니다. 결국 각 feature data와 targer data 사이에 상관관계가 클 수록 target 값 예측에 유리한데, 각 데이터 사이의 상관관계를 시각화 하면 다음과 같습니다. 

<div align="center">
  
![image](https://github.com/user-attachments/assets/a06f0e46-e97a-477e-b873-01a7e8b4d436)

<p style="margin-top: 10px;">Figure 4. Heatmap of Total Data/p>
</div>

Heatmap을 통하여 각 데이터 사이의 상관관계를 한눈에 파악할 수 있으며, 이를 수치화 하면 다음과 같습니다.
```
Targets와 Features 간의 상관관계:
Shell_weight      0.627574
Diameter          0.574660
Height            0.557467
Length            0.556720
Whole_weight      0.540390
Viscera_weight    0.503819
Shucked_weight    0.420884
Name: Rings, dtype: float64
```

### 2. Preprocessing
#### 2.1. Encoding Non-Numerical Data
우선 딥러닝 모델의 학습을 위하여 Non-Numerical Feature Data인 Sex data를 수치화시키면 다음과 같습니다.
```
Length  Diameter  Height  Whole_weight  Shucked_weight  Viscera_weight  \
0   0.455     0.365   0.095        0.5140          0.2245          0.1010   
1   0.350     0.265   0.090        0.2255          0.0995          0.0485   
2   0.530     0.420   0.135        0.6770          0.2565          0.1415   
3   0.440     0.365   0.125        0.5160          0.2155          0.1140   
4   0.330     0.255   0.080        0.2050          0.0895          0.0395   

   Shell_weight  Sex_F  Sex_I  Sex_M  
0         0.150      0      0      1  
1         0.070      0      0      1  
2         0.210      1      0      0  
3         0.155      0      0      1  
4         0.055      0      1      0
```

#### 2.2. Removing Data
모든 feature data를 수치화 한 뒤에 본격적으로 데이터 전처리를 시작합니다. 우선 모든 데이터를 처리하기에 앞서서 이상치를 제거해줍니다.
<div align="center">
  
![image](https://github.com/user-attachments/assets/baeb6694-7b12-435f-8224-7796b247fe64)
<p style="margin-top: 10px;">Figure 5. Outlier of Feature Data</p>
</div>

<div align="center">
  
![image](https://github.com/user-attachments/assets/822b038a-106f-4214-bd2d-c0684a67e597)
<p style="margin-top: 10px;">Figure 6. Remove Outlier</p>
</div>

이상치 제거의 경우 필요한 정보를 담고 있는 데이터 또한 제거할 수 있기 때문에 그 범위를 잘 지정하여야 합니다. 다만 데이터를 수집하는 동안 발생할 수 있는 오류를 배제하기 위하여 이상치 제거를 선택하였습니다. 

또한 평균 값으로부터 일정 수준 이상 벗어나지 않더라도 물리적으로 성립되지 않는 데이터가 존재할 수 있습니다. 이번 과제에서는 총 두가지를 고려하였습니다. 첫번째로 전체 전복 무게가 '살만 포함한 무게 + 껍질 무게' 보다 작은 경우입니다. 전체 전복 무게의 경우 '전복 살 무게 + 껍질 무게' 뿐만 아니라 내장 무게를 포함하고 있기 때문에 첫번째에 해당하는 데이터는 잘못 수집된 경우일 것입니다. 두번째로 전복 살 무게가 
