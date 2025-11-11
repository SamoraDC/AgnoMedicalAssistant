"""OCR processor with PyMuPDF and Tesseract fallback."""
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_settings
from src.core.exceptions import OCRError


class OCRResult(BaseModel):
    """Result from OCR processing."""

    text: str = Field(description="Extracted text")
    page_count: int = Field(description="Number of pages processed")
    method: Literal["pymupdf", "tesseract"] = Field(description="OCR method used")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="OCR confidence")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class OCRProcessor:
    """OCR processor for extracting text from documents.

    Features:
    - Primary: PyMuPDF for digital PDFs
    - Fallback: Tesseract for scanned documents/images
    - Automatic method selection
    - Multi-page support
    """

    def __init__(self):
        """Initialize OCR processor."""
        self.settings = get_settings()
        self._setup_tesseract()

    def _setup_tesseract(self) -> None:
        """Configure Tesseract OCR if needed."""
        if self.settings.tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd
                logger.info(f"Tesseract configured: {self.settings.tesseract_cmd}")
            except ImportError:
                logger.warning("pytesseract not installed, Tesseract OCR unavailable")

    def _extract_with_pymupdf(self, file_path: Path) -> OCRResult:
        """Extract text using PyMuPDF (for digital PDFs).

        Args:
            file_path: Path to PDF file

        Returns:
            OCR result with extracted text

        Raises:
            OCRError: If extraction fails
        """
        try:
            logger.info(f"Extracting text with PyMuPDF from {file_path.name}")

            doc = fitz.open(file_path)
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            full_text = "\n\n".join(text_parts)

            # Check if text was extracted
            if not full_text.strip():
                logger.warning("No text extracted with PyMuPDF, may need Tesseract")
                raise OCRError("No text content found in PDF")

            doc.close()

            return OCRResult(
                text=full_text,
                page_count=len(doc),
                method="pymupdf",
                metadata={"file_name": file_path.name}
            )

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise OCRError(
                f"Failed to extract text with PyMuPDF",
                details={"error": str(e), "file": str(file_path)}
            )

    def _extract_with_tesseract(self, file_path: Path) -> OCRResult:
        """Extract text using Tesseract OCR (for scanned documents).

        Args:
            file_path: Path to image or PDF file

        Returns:
            OCR result with extracted text

        Raises:
            OCRError: If extraction fails
        """
        try:
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path

            logger.info(f"Extracting text with Tesseract from {file_path.name}")

            # Convert PDF to images if needed
            if file_path.suffix.lower() == '.pdf':
                images = convert_from_path(file_path)
            else:
                images = [Image.open(file_path)]

            text_parts = []
            total_confidence = 0.0

            for i, image in enumerate(images):
                # Extract text with confidence data
                ocr_data = pytesseract.image_to_data(
                    image,
                    lang=self.settings.ocr_languages,
                    output_type=pytesseract.Output.DICT
                )

                # Calculate page confidence
                confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
                page_confidence = sum(confidences) / len(confidences) if confidences else 0
                total_confidence += page_confidence

                # Extract text
                text = pytesseract.image_to_string(
                    image,
                    lang=self.settings.ocr_languages
                )
                text_parts.append(f"--- Page {i + 1} ---\n{text}")

            full_text = "\n\n".join(text_parts)
            avg_confidence = total_confidence / len(images) if images else 0

            return OCRResult(
                text=full_text,
                page_count=len(images),
                method="tesseract",
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                metadata={
                    "file_name": file_path.name,
                    "languages": self.settings.ocr_languages
                }
            )

        except ImportError as e:
            logger.error(f"Tesseract dependencies not installed: {e}")
            raise OCRError(
                "Tesseract OCR not available - install pytesseract and pdf2image",
                details={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise OCRError(
                f"Failed to extract text with Tesseract",
                details={"error": str(e), "file": str(file_path)}
            )

    def process(
        self,
        file_path: str | Path,
        force_method: Literal["pymupdf", "tesseract"] | None = None
    ) -> OCRResult:
        """Process document and extract text.

        Args:
            file_path: Path to document file
            force_method: Force specific OCR method (optional)

        Returns:
            OCR result with extracted text

        Raises:
            OCRError: If processing fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise OCRError(
                f"File not found: {file_path}",
                details={"file": str(file_path)}
            )

        logger.info(f"Processing document: {file_path.name}")

        # Determine method
        if force_method:
            method = force_method
        elif self.settings.ocr_engine == "tesseract":
            method = "tesseract"
        else:
            method = "pymupdf"

        try:
            # Try primary method
            if method == "pymupdf":
                try:
                    return self._extract_with_pymupdf(file_path)
                except OCRError:
                    # Fallback to Tesseract if PyMuPDF fails
                    logger.info("Falling back to Tesseract OCR")
                    return self._extract_with_tesseract(file_path)
            else:
                return self._extract_with_tesseract(file_path)

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise OCRError(
                f"Failed to process document",
                details={"error": str(e), "file": str(file_path)}
            )

    async def process_async(
        self,
        file_path: str | Path,
        force_method: Literal["pymupdf", "tesseract"] | None = None
    ) -> OCRResult:
        """Async wrapper for OCR processing.

        Args:
            file_path: Path to document file
            force_method: Force specific OCR method (optional)

        Returns:
            OCR result with extracted text
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process,
            file_path,
            force_method
        )
