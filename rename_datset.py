import os
import re

def rename_images(directory):
    # 디렉토리 내의 모든 파일과 폴더를 순회
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Image_숫자 패턴을 찾음
            match = re.search(r'Image_(\d+)', file)
            if match:
                old_num = int(match.group(1))
                # 1~60 범위 내의 숫자만 처리
                if 1 <= old_num <= 60:
                    new_num = old_num + 120
                    # 새 파일명 생성
                    new_file = file.replace(f'Image_{old_num}', f'Image_{new_num}')
                    # 전체 경로 구성
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_file)
                    # 파일명 변경
                    os.rename(old_path, new_path)
                    print(f'Renamed: {old_path} -> {new_path}')

# 사용 예:
if __name__ == "__main__":
    folder_path = "korean_flowers_dataset_bak"  # 변경하려는 폴더 경로
    rename_images(folder_path)