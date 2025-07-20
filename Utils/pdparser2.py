import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from datetime import datetime
import sys

class SimplePDFTextExtractor:
    def __init__(self):
        pass

    def extract_and_clean_text(self, pdf_path: str) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists() or not pdf_path.suffix.lower() == '.pdf':
            raise FileNotFoundError(f"Invalid PDF path: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        raw_text = " ".join(page.get_text("text") for page in doc)  # Join all pages into one string
        doc.close()
        
        return self._clean_text(raw_text)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)  # Replace all whitespace (\n, \t, etc.) with single space
        text = re.sub(r'[^\w\s@.\-+]', '', text)  # Remove all special characters except essential ones
        return text.strip()

    def save_text_to_json(self, text: str, pdf_path: str, output_path: str = None) -> str:
        pdf_file = Path(pdf_path)
        output_file = Path(output_path) if output_path else pdf_file.with_name(pdf_file.stem + "_raw_text.json")
        
        data = {
            "filename": pdf_file.name,
            "extracted_on": datetime.now().isoformat(),
            "raw_text": text
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(output_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        extractor = SimplePDFTextExtractor()
        print(f"📄 Extracting and compressing text from: {pdf_path}")
        cleaned_text = extractor.extract_and_clean_text(pdf_path)
        output_path = extractor.save_text_to_json(cleaned_text, pdf_path)
        print(f"✅ Saved optimized raw text to JSON: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
