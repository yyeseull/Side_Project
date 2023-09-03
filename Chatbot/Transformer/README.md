# Transformer Chatbot 

## 데이터 출처 
 [AI 허브](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86)

***

 ## 프로젝트 리포트
 [노션 링크](https://peaceful-boat-1d8.notion.site/0892712a10094091b75e7123ad3146e0?pvs=4)

***


`토크나이저` : Subword Text Encoder(서브워드 텍스트 인코더)
`손실함수` : Sparse Categorical Cross Entropy
`옵티마이저` : Adam
`평가지표` : sparse categorical accuracy


***

## 주요 하이퍼파리미터 

- $d_{model}$ = 256
- num_layers = 3
- num_heads = 8
- $d_{ff}$ = 512
- Dropout = 0.1

***


## 최종 결과 

![initial]("https://github.com/yyeseull/Side_Project/assets/102211628/c1b7a2f6-4567-4c1c-a7af-bc8222157fe3")

![initial]("https://github.com/yyeseull/Side_Project/assets/102211628/1c56bc1b-cb7e-4ca5-9405-72ab76b54d52")
