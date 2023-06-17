# 제주도 도로 교통량 예측 AI 경진대회  

## 데이터 

- 출처
[데이콘 주소](https://dacon.io/competitions/official/235985/overview/description)

- 외부 데이터 출처  
[기상청 기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)


***


## 프로젝트 정리 
- [노션 링크](https://peaceful-boat-1d8.notion.site/0892712a10094091b75e7123ad3146e0?pvs=4) 

***


## 모델 
- `평가 `: MAE
- `LGBMRegressor `-> 5.227474509459903
- `XGboost` -> 4.663731015551993
- `RandomForestRegressor` -> 4.537415594989837
- `Catboost` -> 4.560542796092552 

***


### 최종 모델 
`Catboost Hyperparameter tuning`
- public 점수 :  `4.3783928159`/  private 점수 : `4.371353451`
