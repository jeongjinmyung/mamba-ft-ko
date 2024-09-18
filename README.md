# Mamba Korean Finetuning

Mamba 모델 한국어 파인튜닝 스크립트

### Data
`HAERAE-HUB/KOREAN-WEBTEXT` 데이터셋 사용

### Model
`state-spaces/mamba-130m-hf` 모델 사용

### Peft
Lora 적용

### Device
V 100 2장 사용하여 약 150시간 학습


### Training
requirements.txt 파일 안의 dependencies 설치 필요

```bash
pip install -r requirements.txt
```

설치 후 다음 쉘 스크립트 실행

```bash
./scripts/test.sh
```

### Issues
- 허깅페이스 허브에 올라온 자료들마다 variation이 있음\
사용하는 모델과 데이터에 맞게 코드를 일부 수정해야 할 수 있음

### To-Do (모델 사이즈가 작아 일부는 한계가 있을 것으로 보임)
- 허깅페이스에 모델 올리기
- Korean Instruction Tuning
- RAG 적용 및 개인화
- 웹 인터페이스 제작
