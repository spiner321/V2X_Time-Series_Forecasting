# 신규 파일 다운로드
- 50-1 신규 파일 다운로드는 data/update/zip 폴더에 저장
- 이후 `unzip_raw.py` 파일을 실행하면 자동으로 uncombine, combine에 나누어 csv로 저장됨

# 모델 추론
- `python v2x_test.py --model_name {turn, speed} --average {macro, micro, weighted} --continue_test {result.txt 파일 경로}`

## 1. 모델 전처리
- `V2X_preprocess_8 ~ 11.ipynb` 파일을 참고

## 2. 모델 학습, 추론 및 결과 저장
- `V2X_inference_turn.ipynb`, `V2X_inference_hazard` 파일을 참고

## 3. 모델 통계
- `V2X_statistics.ipynb` 파일을 참고