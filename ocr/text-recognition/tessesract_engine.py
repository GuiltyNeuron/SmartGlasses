import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\tesseract-ocr\tesseract\bin\tesseract.exe'

img = cv2.imread("./data/test.png")

text = pytesseract.image_to_string(img, lang = 'eng')

print(text)