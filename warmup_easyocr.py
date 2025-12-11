import easyocr

print("Initializing EasyOCR (this may download models the first time)...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR is ready.")
