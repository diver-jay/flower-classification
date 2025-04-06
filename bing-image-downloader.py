import os
import time
from bing_image_downloader import downloader
import random

# 한국 플로리스트가 자주 사용하는 대표 꽃 10종류
korean_flowers = [
    "장미", # Rose
    "튤립", # Tulip
    "국화", # Chrysanthemum
    "카네이션", # Carnation
    "백합", # Lily
    "프리지아", # Freesia
    "거베라", # Gerbera
    "수국", # Hydrangea
    "작약", # Peony
    "라넌큘러스" # Ranunculus
]

# 각 꽃당 최대 다운로드 이미지 수
max_images_per_flower = 60

# 이미지를 저장할 기본 디렉토리
output_dir = "korean_flowers_dataset"

# 다운로드 실행 함수
def download_flower_images():
    print("한국 대표 꽃 이미지 다운로드를 시작합니다...")
    
    for flower in korean_flowers:
        print(f"\n다운로드 시작: {flower}")
        query = f"{flower} 꽃"
        
        try:
            # bing_image_downloader를 사용하여 이미지 다운로드
            downloader.download(
                query,
                limit=max_images_per_flower,
                output_dir=output_dir,
                adult_filter_off=False,
                force_replace=False,
                timeout=60,
                verbose=True
            )
            
            print(f"{flower} - 다운로드 완료")
            
            # 과도한 요청 방지를 위한 대기 시간
            wait_time = random.uniform(1.0, 3.0)
            print(f"{wait_time:.1f}초 대기 중...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"다운로드 중 오류 발생: {e}")
    
    print("\n모든 꽃 이미지 다운로드 완료!")
    print(f"다운로드 경로: {os.path.abspath(output_dir)}")

    # 다운로드된 이미지 수 확인
    total_images = 0
    for flower in korean_flowers:
        query = f"{flower} 꽃"
        flower_dir = os.path.join(output_dir, query)
        if os.path.exists(flower_dir):
            image_count = len([f for f in os.listdir(flower_dir) if os.path.isfile(os.path.join(flower_dir, f))])
            print(f"{flower}: {image_count}장")
            total_images += image_count
    
    print(f"총 다운로드된 이미지: {total_images}장")

# 메인 실행 코드
if __name__ == "__main__":
    download_flower_images()