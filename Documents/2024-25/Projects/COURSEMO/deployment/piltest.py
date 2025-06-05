from PIL import Image
from io import BytesIO
import base64
import cv2
import pytesseract
import numpy as np

def preprocess(img, kernel_size=60): 
   # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Approximate background using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Subtract background and normalize
    shadow_free = cv2.subtract(background, gray)
    shadow_free = cv2.normalize(shadow_free, None, 0, 255, cv2.NORM_MINMAX)
    return shadow_free

with open('./deployment/image5.jpg', "rb") as image_file:
    imagebytes = base64.b64encode(image_file.read()).decode('utf-8')
    img_bytes = base64.b64decode(imagebytes)
    import hashlib
    print(hashlib.md5(img_bytes).hexdigest())  # Before processing
    print(pytesseract.get_tesseract_version())
    array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    processed = preprocess(image)
    PIL_image = Image.fromarray(np.uint8(processed))
    PIL_image.show()
    config = r'--psm 3 --oem 3'
print(pytesseract.image_to_string(processed, lang='eng', config=config))
