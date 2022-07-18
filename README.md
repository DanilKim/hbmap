# Hbmap Segmentation Project
PyTorch deep learning project made easy.

## Usage


### Docker 환경 구축
`onlyone21/tree_hpa:latest` 이미지로부터 container 생성.

`컨테이너 이름` 과 `코드 경로 -> 마운트 경로` 만 설정해주면 됨.

  ```
  docker run -it --ipc=host --gpus=all -v (코드 폴더 위치):(마운트할 디렉토리) -v /disk3/yunseung/hbmap:/data --name=(컨테이너 이름) onlyone21/tree_hpa:latest /bin/bash
  ```

### 학습 돌리기
학습 관련 모든 설정들은 `config.json` 에서 할 수 있음.

`-d` 통해서 gpu 번호 지정 가능. gpu 여러개 쓸 경우, `,`로 이어 붙이면 됨.

  ```
  # Single-GPU
  python train.py -d 0 -c config.json

  # Multi-GPU
  python train.py -d 0,1,5 -c config.json 
  ```
GPU 고정시,고정된 GPU의 번호 순서대로 0,1,2, .. 로 다시 매겨짐
  ```
  # 아래의 경우 실제로는 GPU 3,5 를 사용하게 됨.
  CUDA_VISIBLE_DEVICES=3,4,5 python train.py -d 0,2 -c config.json 
  ```

