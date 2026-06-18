import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from docx import Document


class DocumentLoader:

    def load_txt(self, file_path):

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return {
            "text": text,
            "source": Path(file_path).name
        }

    def load_pdf(self, file_path):

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return {
            "text": text,
            "source": Path(file_path).name
        }

    def load_docx(self, file_path):

        doc = Document(file_path)

        text = "\n".join(
            [paragraph.text for paragraph in doc.paragraphs]
        )

        return {
            "text": text,
            "source": Path(file_path).name
        }

    def load_csv(self, file_path):

        df = pd.read_csv(file_path)

        text = df.to_string(index=False)

        return {
            "text": text,
            "source": Path(file_path).name
        }

    def load_document(self, file_path):

        extension = Path(file_path).suffix.lower()

        if extension == ".txt":
            return self.load_txt(file_path)

        elif extension == ".pdf":
            return self.load_pdf(file_path)

        elif extension == ".docx":
            return self.load_docx(file_path)

        elif extension == ".csv":
            return self.load_csv(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )


if __name__ == "__main__":

    loader = DocumentLoader()

    document = loader.load_document("data/sample.csv")

    print(document["source"])
    print(document["text"][:500])