import os
import requests

# File info
FILE_ID = "115b1K2j7DGb7UXEVUrPlAoZn9fsmIA3H"
FILE_NAME = "yolov3.weights"

def download_from_google_drive(file_id, destination):
    print("Starting download from Google Drive...")
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle large file warning
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"{destination} downloaded successfully.")

# Check if already downloaded
if not os.path.exists(FILE_NAME):
    print(f"{FILE_NAME} not found. Downloading...")
    download_from_google_drive(FILE_ID, FILE_NAME)
else:
    print(f"{FILE_NAME} already exists.")
