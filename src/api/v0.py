import time
from fastapi import APIRouter, UploadFile, HTTPException, Form, Depends, File
from typing import List
from ..abstracting.keywords_abstracting import extract_keywords
from ..abstracting.classic_abstract import TextSummarizer
from ..recognition.controller import resolve, RecognitionMethod
from ..abstracting.neural_abstract import BilingualSummarizer
from ..utils import load_documents_and_languages


def get_summarizer() -> TextSummarizer:
    documents, languages = load_documents_and_languages()
    return TextSummarizer(documents, languages)

def get_mbart_summarizer() -> BilingualSummarizer:
    return BilingualSummarizer()

def create_router() -> APIRouter:
    query_router = APIRouter()

    @query_router.post("/upload-html/")
    async def query(files: List[UploadFile] = File(...), 
                    method: RecognitionMethod = Form(...),
                    summarizer: TextSummarizer = Depends(get_summarizer),
                    bilingual_summarizer: BilingualSummarizer = Depends(get_mbart_summarizer)):
        results = []

        for file in files:
            if file.content_type != "text/html":
                raise HTTPException(status_code=400, detail=f"Неверный формат файла: {file.filename}. Ожидается HTML.")
            
            start_time = time.perf_counter()
            language, extracted_text = await resolve(file, method)
            extraction_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            keywords_summary = extract_keywords(extracted_text, language)
            keywords_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            classic_summary = summarizer.summarize(extracted_text, language)
            classic_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            neural_summary = bilingual_summarizer.summarize_text(extracted_text, language)
            neural_time = time.perf_counter() - start_time

            results.append({
                "filename": file.filename,
                "language": language,
                "classic_summary": classic_summary,
                "keywords_summary": keywords_summary,
                "neural_summary": neural_summary,
                "times": {
                    "extraction_time": extraction_time,
                    "keywords_time": keywords_time,
                    "classic_time": classic_time,
                    "neural_time": neural_time
                }
            })

        return {"results": results}

    return query_router
