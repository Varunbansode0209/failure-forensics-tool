#The Summarization step takes everything the previous stages have understood about the document and produces a human-readable summary.

#FLOW
# DOCUMENT - INTAKE --document_type-- - EXTRACTION --extraction_field-- - CLASSIFICATION --category-- - SUMMARIZATION - GEMINI 3.8FLASH - STRUCTURE JSON - PYDANTIC VALIDATION - SUMMARIZATIONOUTPUT




from app.models.pipeline_models import(
    SummarizationInput,
    SummarizationOutput,
)
from app.tracing.decorator import traced

from app.llm.client import call_llm

@traced
def summarization(document: SummarizationInput) -> SummarizationOutput:
    """
    Generate a concise summary and key points from the processed document.

    """

    prompt = f"""
You are the summarization stage of an AI document processing pipeline.

Create a concise summary of the document using the information
provided by the previous pipeline stages.

Document type:
{document.document_type}

Category:
{document.category}

Extracted fields:
{document.extracted_fields}

Original document:
{document.document_text}

Return ONLY valid JSON using exactly this structure:

{{
    "summary": "concise summary of the document",
    "key_points": [
        "important point 1",
        "important point 2"
    ],
    "confidence": 0.0
}}

Rules:
- Use only information present in the document.
- Do not invent facts.
- Keep the summary concise.
- Include the most important facts in key_points.
- Confidence must be between 0.0 and 1.0.
"""


    llm_result = call_llm(prompt)
    result = llm_result.text

    return SummarizationOutput.model_validate_json(result)