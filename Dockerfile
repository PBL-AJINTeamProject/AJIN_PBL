# 베이스 이미지
FROM python:3.9-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# 작업 디렉토리 설정
WORKDIR /workspace

# 프로젝트 파일 복사
COPY . .

# Python 의존성 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install ultralytics

CMD ["python", "AI HDS.py"]
