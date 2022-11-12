from doctr.io import DocumentFile
from doctr.models import ocr_predictor




model = ocr_predictor(pretrained=True)
# PDF
pdf_doc = DocumentFile.from_pdf("test.pdf")
# Image
# single_img_doc = DocumentFile.from_images("path/to/your/img.jpg")


# Analyze
result = model(pdf_doc)

print()