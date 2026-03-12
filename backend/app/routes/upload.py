import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Request, UploadFile, File
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import UploadResponse, UploadResult
from app.services.pdf_processor import PDFProcessor
from app.services.metadata_extractor import MetadataExtractor

logger = logging.getLogger("cv_analyzer.routes.upload")
router = APIRouter()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

pdf_processor = PDFProcessor()


@router.post("/upload", response_model=UploadResponse)
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    vector_store = request.app.state.vector_store
    llm_service = request.app.state.llm_service
    extractor = MetadataExtractor(llm_service)

    results: list[UploadResult] = []

    for file in files:
        try:
            content = await file.read()

            # Validate
            valid, error_msg = pdf_processor.validate_file(content, file.filename or "unknown")
            if not valid:
                results.append(UploadResult(filename=file.filename or "unknown", status="error", message=error_msg))
                continue

            # Check duplicate
            file_hash = pdf_processor.compute_hash(content)
            if vector_store.check_hash_exists(file_hash):
                results.append(UploadResult(filename=file.filename or "unknown", status="duplicate", message="File already exists in database"))
                continue

            # Extract text
            success, text = pdf_processor.extract_text(content)
            if not success:
                results.append(UploadResult(filename=file.filename or "unknown", status="error", message=text))
                continue

            # Extract metadata via LLM
            metadata = await extractor.extract(text)

            # Generate CV ID
            cv_id = str(uuid.uuid4())

            # Chunk text
            chunks = text_splitter.split_text(text)

            # Add metadata
            metadata["file_hash"] = file_hash
            metadata["filename"] = file.filename or "unknown"
            metadata["upload_date"] = datetime.now(timezone.utc).isoformat()

            # Store in vector DB
            vector_store.add_cv(
                cv_id=cv_id,
                candidate_name=metadata["candidate_name"],
                full_text=text,
                chunks=chunks,
                metadata=metadata,
            )

            # Save original PDF
            pdf_processor.save_file(content, file.filename or f"{cv_id}.pdf")

            results.append(UploadResult(
                filename=file.filename or "unknown",
                status="success",
                cv_id=cv_id,
                message=f"Processed successfully: {metadata['candidate_name']}",
            ))
            logger.info("Uploaded CV: %s (%s)", metadata["candidate_name"], cv_id)

        except Exception as e:
            logger.error("Error processing %s: %s", file.filename, e, exc_info=True)
            results.append(UploadResult(filename=file.filename or "unknown", status="error", message=str(e)))

    return UploadResponse(success=True, results=results)
