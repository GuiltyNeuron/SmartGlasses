import pytesseract


class TesseractEngine:

    def __init__(self):

        self.tessdata_dir_config = r'--tessdata-dir "ocr_api\tessdata"'
        self.tesseract_exec_path = r'C:\tesseract\tesseract\bin\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_exec_path

    def img2txt(self, img, language):
        return pytesseract.image_to_string(img, lang = language, config=self.tessdata_dir_config)