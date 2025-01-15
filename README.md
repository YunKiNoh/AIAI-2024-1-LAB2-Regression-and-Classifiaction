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

또한 평균 값으로부터 일정 수준 이상 벗어나지 않더라도 물리적으로 성립되지 않는 데이터가 존재할 수 있습니다. 이번 과제에서는 총 두가지를 고려하였습니다. 첫번째로 전체 전복 무게가 '살만 포함한 무게 + 껍질 무게' 보다 작은 경우입니다. 전체 전복 무게의 경우 '전복 살 무게 + 껍질 무게' 뿐만 아니라 내장 무게를 포함하고 있기 때문에 첫번째에 해당하는 데이터는 잘못 수집된 경우일 것입니다. 두번째로 전복 살 무게가 내장 무게 보다 작은 경우입니다. 이는 전복의 살이 내장보다 차지하는 비율이 높기 때문에 두번째 경우에 해당하는 데이터셋은 배제하였습니다.

#### 2.3. Normalization
각각의 feature data를 학습에 그대로 사용할 경우 target data의 최댓값이 클 수록 예측에 미치는 영향이 커지기 때문에 정확한 예측을 하기가 어려워집니다. 이를 피하기 위하여 모든 feature data를 정규화하여 각 데이터에 대한 가중치를 적절히 선정할 수 있도록 합니다.

```
Length  Diameter    Height  Whole_weight  Shucked_weight  \
0     0.513514  0.521008  0.084071      0.181335        0.150303   
1     0.371622  0.352941  0.079646      0.079157        0.066241   
2     0.614865  0.613445  0.119469      0.239065        0.171822   
3     0.493243  0.521008  0.110619      0.182044        0.144250   
4     0.344595  0.336134  0.070796      0.071897        0.059516   
...        ...       ...       ...           ...             ...   
4172  0.662162  0.663866  0.146018      0.313441        0.248151   
4173  0.695946  0.647059  0.119469      0.341420        0.294553   
4174  0.709459  0.705882  0.181416      0.415796        0.352724   
4175  0.743243  0.722689  0.132743      0.386931        0.356422   
4176  0.858108  0.840336  0.172566      0.689393        0.635171   

      Viscera_weight  Shell_weight  Sex_F  Sex_I  Sex_M  
0           0.132324      0.147982    0.0    0.0    1.0  
1           0.063199      0.068261    0.0    0.0    1.0  
2           0.185648      0.207773    1.0    0.0    0.0  
3           0.149440      0.152965    0.0    0.0    1.0  
4           0.051350      0.053313    0.0    1.0    0.0  
...              ...           ...    ...    ...    ...  
4172        0.314022      0.246637    1.0    0.0    0.0  
4173        0.281764      0.258097    0.0    0.0    1.0  
4174        0.377880      0.305431    0.0    0.0    1.0  
4175        0.342989      0.293473    1.0    0.0    0.0  
4176        0.495063      0.491779    0.0    0.0    1.0  

[4177 rows x 10 columns]
```

#### 2.4. Divide Train and Test dataset
최종적으로 데이터를 학습 및 검증에 사용하기 위하여 적절히 분할해줍니다. 해당 과제의 경우 test:validation의 비율을 2:8로 분할하였습니다.

### 3. Train Deep Learning Model
이번 경우에서는 회귀 모델 중 높은 성능을 보여주는 RndomForest와 GradientBoosting을 Ensemble하여 학습을 진행하였습니다. RandomForest의 경우 Bagging(Bootstrap Aggregation)을 통하여 모델을 학습시키는 방법으로, 하나의 학습 데이터를 여러갈래로 나뉘어서 여러번 모델을 학습시키기 때문에 학습에 있어서 효율적으로 높은 예측 결과를 도출할 수 있습니다. GradientBoosting의 경우 Bagging과는 다른 앙상블 기법 모델로써, 학습 데이터를 분할하여 단계별로 모델을 잔차를 줄여나가는 학습 방식입니다. 서로 다른 두 모델을 통하여 더 높은 정확도를 도출해내기 위하여 앙상블 시켜 학습을 진행하였습니다. 그 결과는 다음과 같습니다.
```
앙상블 모델 결과:
MAE: 1.5053979037509384
RMSE: 2.133168007407893
```

모델 예측의 정확도와 더불어 중요하게 평가되는 요소는 과적합 여부입니다. 아무리 학습 데이터에 대하여 예측이 정확하더라도 다른 데이터 셋에서 높은 정확도를 보여주지 못한다면 해당 모델은 정확도가 낮은 것이기 때문입니다. 이번 과제에서 학습 데이터에 대한 Error가 검증 데이터에 대한 Error보다 항상 높기 때문에 과적합되지 않았다고 판단하였습니다.
<div align="center">
  
![image](https://github.com/user-attachments/assets/016ac72c-6095-4339-9763-1d15d6ff8682)
<p style="margin-top: 10px;">Figure 7. Learning Curve</p>
</div>

## Part2: Predict Obesity Level by Using Classification Model
### 2.1. Dataset Information
Part2에서는 14세에서 61세의 멕시코, 페루, 그리고 콜롬비아 사람들에 대한 신체 정보를 활용하여 비만의 위험성 정도를 예측하였습니다. 
<div align="center">

| **Attribute Name**      | **Category**         | **Description**                                        | **Data Type**       | **Units**        |
| :----------------------- | :------------------- | :----------------------------------------------------- | :------------------ | :--------------- |
| **Gender**              | Demographics         | Gender of the individual (Male/Female)                | Categorical         | -                |
| **Age**                 | Demographics         | Age of the individual                                  | Continuous          | Years            |
| **Height**              | Demographics         | Height of the individual                               | Continuous          | Meters           |
| **Weight**              | Demographics         | Weight of the individual                               | Continuous          | Kilograms        |
| **FAVC**                | Eating Habits        | Frequent consumption of high-caloric food (Yes/No)    | Categorical         | -                |
| **FCVC**                | Eating Habits        | Frequency of vegetable consumption                    | Continuous          | -                |
| **NCP**                 | Eating Habits        | Number of main meals per day                          | Integer             | Count            |
| **CAEC**                | Eating Habits        | Consumption of food between meals                     | Categorical         | -                |
| **CH20**                | Eating Habits        | Daily water consumption                               | Continuous          | Liters           |
| **CALC**                | Eating Habits        | Frequency of alcohol consumption                      | Categorical         | -                |
| **SCC**                 | Physical Condition   | Monitoring of calorie consumption (Yes/No)           | Categorical         | -                |
| **FAF**                 | Physical Condition   | Frequency of physical activity                        | Continuous          | Hours/Week       |
| **TUE**                 | Physical Condition   | Time spent using technology devices                   | Continuous          | Hours/Day        |
| **MTRANS**              | Physical Condition   | Main mode of transportation                           | Categorical         | -                |
| **Obesity Levels**      | Target               | BMI-based obesity classification                      | Categorical         | -                |

</div>

<div align="left">
  
| **Obesity Levels (BMI)** | **Range**           |
| :----------------------- | :------------------ |
| Underweight              | Less than 18.5     |
| Normal                  | 18.5 to 24.9       |
| Overweight              | 25.0 to 29.9       |
| Obesity I               | 30.0 to 34.9       |
| Obesity II              | 35.0 to 39.9       |
| Obesity III             | 40 and above       |

<div align="left">

<p style="margin-top: 10px;">Table 1. Obesity Dataset Description</p>

</div>
