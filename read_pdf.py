import fitz  # PyMuPDF
import sys

def main():
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        doc = fitz.open("Smart Voice-controlled Car – Project Documentation.pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            f.write(f"--- Page {page_num + 1} ---\n")
            f.write(page.get_text("text"))

if __name__ == "__main__":
    main()
