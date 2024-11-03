#!/bin/bash
# 시스템 종속성 설치
sudo apt update
sudo apt install -y llvm portaudio19-dev

# 파이썬 패키지 설치
pip install -r requirements.txt

sudo apt-get update
sudo apt-get install -y portaudio19-dev

sudo apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev

sudo apt-get install -y build-essential
