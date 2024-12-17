import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from PIL import Image
#import os
import shutil

import requests
#import os
import sys

def Detect_sign(original_img):
    endpoint = "https://5a030comvision.cognitiveservices.azure.com/"
    key = "C8Egslzv6b8fTXJIlcl0JNgPDNkpqGPGbeBONTzCFAycNP7tDcSBJQQJ99ALACYeBjFXJ3w3AAAFACOG5kUk"

    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Load image to analyze into a 'bytes' object
    with open(original_img, "rb") as f:
        image_data = f.read()

    visual_features =[ #Azure AI Vision 모든 기능 사용 가능
            VisualFeatures.OBJECTS,
            VisualFeatures.SMART_CROPS
        ]

    # Analyze all visual features from an image stream. This will be a synchronously (blocking) call.
    result = client.analyze(
        image_data=image_data,
        visual_features=visual_features,
        smart_crops_aspect_ratios=[0.9, 1.33],
        gender_neutral_caption=True,
        language="en"
    )

    # Print all analysis results to the console
    print("Image analysis results:")

    stop_sign = []
    if result.objects is not None:
        for object in result.objects.list:
            tag_name = object.tags[0].name
            if tag_name == "stop sign":
                stop_sign.append(object.bounding_box)

    print("Detection End\n")
    return stop_sign

def Crop_sign(original_img, stop_sign):

    # bounding_boxes 변수에 stop_sign 리스트 할당
    bounding_boxes = [bbox for bbox in stop_sign]

    # 출력 디렉토리 설정
    output_directory = "cropped_images"

    # 출력 디렉토리가 존재하면 내용을 비우고, 존재하지 않으면 새로 생성
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # 원본 이미지 열기
    img = Image.open(original_img)

    # 각 bounding box에 대해 이미지 자르기 및 저장
    for i, bbox in enumerate(bounding_boxes):
        cropped_img = img.crop((bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']))
        
        cropped_image_path = os.path.join(output_directory, f"stop_sign_{i + 1}.jpg")
        cropped_img.save(cropped_image_path)
    
    print("Crop End\n")

def Classify_sign():
    # 1. URL 설정
    url = "https://5a030cv-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/bede0097-721f-4019-ab17-030022fb00bd/detect/iterations/preModel/image"
    # 2&3. Headers 설정
    headers = {
        "Prediction-Key": "Eal5VuOQMtWpt3VhIFSTYOVO9NN5mwJiCwDQUUb6zrOmGAyWANx7JQQJ99ALACYeBjFXJ3w3AAAIACOGVJK3",
        "Content-Type": "application/octet-stream"
    }

    # 4. 이미지 파일들이 있는 디렉토리 설정
    image_directory = "cropped_images"

    # 5. 결과를 저장할 리스트
    results = []

    # 6. 디렉토리 내의 모든 이미지 파일 처리
    for image_file in os.listdir(image_directory):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자 확인
            image_path = os.path.join(image_directory, image_file)
            
            # 이미지 파일 읽기
            with open(image_path, "rb") as file:
                image_data = file.read()
            
            # POST 요청 보내기
            response = requests.post(url, headers=headers, data=image_data)
            
            # 결과 확인
            result = response.json()
            
            # 가장 높은 probability를 가진 예측 찾기
            highest_probability_tag = max(result['predictions'], key=lambda x: x['probability'])
            tag_result = highest_probability_tag['tagName']
            
            # 결과 저장
            results.append({
                'file_name': image_file,
                'tag': tag_result,
                'probability': highest_probability_tag['probability']
            })

    # 7. 결과 출력
    for result in results:
        print(f"File: {result['file_name']}, Tag: {result['tag']}, Probability: {result['probability']}")
    
    print("Classification End")

def main():
    img = "Fullimage.jpg"
    stop_sign = Detect_sign(img)
    Crop_sign(img, stop_sign)
    Classify_sign()

if __name__ == "__main__":
    main()
