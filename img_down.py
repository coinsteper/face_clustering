import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from io import BytesIO

# 이미지 저장 폴더 설정
SAVE_FOLDER = "downloaded_faces"

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_unique_filename(folder, base_name, ext):
    """중복되지 않는 파일명 생성"""
    counter = 1
    file_path = os.path.join(folder, f"{base_name}.{ext}")
    while os.path.exists(file_path):  # 파일이 이미 존재하면 숫자를 추가
        file_path = os.path.join(folder, f"{base_name}_{counter}.{ext}")
        counter += 1
    return file_path

def download_image(url, folder, count, name):
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        file_path = get_unique_filename(folder, f"{name}_{count}", "jpg")
        image.save(file_path)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Failed to save image {url}: {e}")

def crawl_thumbnails(name, num_images=20):
    create_folder(SAVE_FOLDER)

    # Selenium 드라이버 설정
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("https://images.google.com")
    time.sleep(2)

    # 검색어 입력
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(name + " 얼굴 사진")
    search_box.send_keys(Keys.RETURN)
    time.sleep(3)

    # 이미지 URL 수집
    image_urls = set()
    while len(image_urls) < num_images:
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img")
        for img in thumbnails:
            src = img.get_attribute("src") or img.get_attribute("data-src")
            if src and "http" in src:
                image_urls.add(src)
            if len(image_urls) >= num_images:
                break
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)

    print(f"Found {len(image_urls)} images.")

    # 이미지 다운로드
    for count, url in enumerate(image_urls):
        download_image(url, SAVE_FOLDER, count + 1, name)
        if count + 1 >= num_images:
            break

    driver.quit()

if __name__ == "__main__":
    person_name = input("이름을 입력하세요: ")
    name_list = {
        "니콜라스 홀트", "에드 스크레인",
        "유재석", "하도영",
        "강동원", "주원",
        "싸이", "이수지", "김고은",
        "김종민", "저스틴 비버",
        "이영표", "조셉 고든 래빗",
        "추미애",
        "제임스 맥어보이",
        "조희봉","정두홍"
        "홍지승", "잇섭",
        "박완규", "강형욱",
        "김동준", "한가인",
        "라이언레이놀즈", "라이언 고슬링"
    }

    for person_name in name_list:
        crawl_thumbnails(person_name, num_images=20)
