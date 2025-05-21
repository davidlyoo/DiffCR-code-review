# 📘 DiffCR 코드 리뷰

본 리포지토리는 [DiffCR](https://github.com/XavierJiezou/DiffCR) 원본 코드에 대한 **심층 코드 리뷰 및 주석 추가 작업**을 다룬 프로젝트입니다.  
DiffCR 프레임워크의 구조와 동작을 정확히 이해하고, 연구 재현 및 확장 가능성을 높이기 위한 문서화 작업이 주된 목적입니다.



## 📌 프로젝트 목적

- DiffCR의 전체 구조와 학습 흐름을 코드 레벨에서 분석
- 주요 코드 파일에 **함수 단위 주석 및 설명 추가**
- 연구 및 구현 기반 확장을 위한 **구조 중심의 문서화**



## 🛠️ 주요 변경 및 추가 내용

 - 코드 주석 및 해설 추가
 - ours_sigmoid.json 기반으로 구성된 네트워크 구조 중심으로 핵심 구조만 정리
 - 불필요한 실험 구성은 제외하고, 주요 네트워크 코드만 유지


---
## Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

To download datasets:

- Sen2_MTC_Old: [multipleImage.tar.gz](https://doi.org/10.7910/DVN/BSETKZ)

- Sen2_MTC_New: [CTGAN.zip](https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view?usp=share_link)

## Training

To train the models in the paper, run these commands:

```train
python run.py -p train -c config/ours_sigmoid.json
```

<!-- >📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters. -->

## Testing

To test the pre-trained models in the paper, run these commands:

```bash
python run.py -p test -c config/ours_sigmoid.json
```

## Evaluation

To evaluate my models on two datasets, run:

```bash
python evaluation/eval.py -s [ground-truth image path] -d [predicted-sample image path]
```

<!-- >📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below). -->

<!-- ## Pre-trained Models

You can download pretrained models here:

- Our awesome model trained on Sen2_MTC_Old: [diffcr_old.pth](/pretrained/diffcr_old.pth)
- Our awesome model trained on Sen2_MTC_New: [diffcr_new.pth](/pretrained/diffcr_new.pth) -->

<!-- >📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models. -->

## 📚 참고 문헌
논문: Diffusion Bridge for Cloud Removal, IEEE TGRS 2024
🔗 https://arxiv.org/abs/2307.16104

원본 코드:
🔗 https://github.com/XavierJiezou/DiffCR
