import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 모델 구조 정의
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # layer
        self.linear1 = nn.Linear(28 * 28, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 10)
        
        # dropout layer
        self.dropout = nn.Dropout(p=0.3)
        
        # activation layer
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

# 모델 불러오기
@st.cache_resource
def load_model():
    # 모델 인스턴스 생성
    model = Classifier()
    # 저장된 가중치 로드
    state_dict = torch.load('fine_tuning.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image):
    # PIL Image를 Tensor로 변환
    transform = transforms.Compose([
        transforms.Grayscale(),  # 흑백으로 변환
        transforms.Resize((28, 28)),  # 크기 조정
        transforms.ToTensor(),  # Tensor로 변환 및 0-1 정규화
        transforms.Lambda(lambda x: 1 - x)  # 이미지 반전 (흰색 글씨 -> 검은 글씨)
    ])
    
    # 이미지 변환 및 배치 차원 추가
    image = transform(image).unsqueeze(0)
    
    return image

def main():
    st.title('손글씨 숫자 인식 시스템')
    
    # 모델 로드
    model = load_model()
    
    # 파일 업로더
    uploaded_file = st.file_uploader("숫자 이미지를 선택하세요", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 이미지', width=200)
        
        # 예측 버튼
        if st.button('숫자 인식하기'):
            # 이미지 전처리
            processed_image = preprocess_image(image)
            
            # 예측
            with torch.no_grad():
                output = model(processed_image)
                probabilities = F.softmax(output, dim=1)
                predicted_number = torch.argmax(probabilities).item()
                confidence = float(probabilities[0][predicted_number])
            
            # 결과 표시
            col1, col2 = st.columns(2)
            with col1:
                st.write('예측된 숫자:')
                st.title(f'{predicted_number}')
            with col2:
                st.write('신뢰도:')
                st.title(f'{confidence:.2%}')

    st.markdown("""
    ### 사용 방법
    1. '숫자 이미지를 선택하세요' 버튼을 클릭하여 이미지를 업로드하세요
    2. '숫자 인식하기' 버튼을 클릭하세요
    3. 인식 결과와 신뢰도를 확인하세요
    
    ### 주의사항
    - 깨끗한 손글씨 이미지를 사용해주세요
    - 하나의 숫자만 포함된 이미지를 사용해주세요
    - 지원 형식: PNG, JPG, JPEG
    """)

if __name__ == '__main__':
    main()