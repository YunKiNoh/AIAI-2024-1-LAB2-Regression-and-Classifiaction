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

<p style="margin-top: 10px;">Figure 1. Categorical feature</p>
</div>

<div align="center">
  
![image](https://github.com/user-attachments/assets/fac4a2ce-0c81-4dce-8dbd-2658f14bf296)

<p style="margin-top: 10px;">Figure 2. Numerical feature</p>
</div>
