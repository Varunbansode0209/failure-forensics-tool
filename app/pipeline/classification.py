#The purpose of classification.py is to take the document text + extracted fields and determine which business category the document belongs to.

#FLOW 
# Intake - Extraction --document_text + extracted_field - classification - gemini - category+reasoning_confidence - Pydantic validation - ClassificationOutput




from app.models.pipeline_models import(ClassificationInput,ClassificationOutput)
from app.tracing.decorator import traced
from app.llm.client import call_llm
@traced
def classification(document: ClassificationInput)-> ClassificationOutput:
    """
    Classify a document using its text and extracted fields.
    """

    prompt = f"""
You are the classification stage of an AI document processing pipeline.

Classify the document into exactly one of these categories:

- legal
- financial
- operational
- general

Document:
{document.document_text}

Extracted fields:
{document.extracted_fields}

Return ONLY valid JSON using exactly this structure:

{{
    "category": "legal | financial | operational | general",
    "confidence": 0.0,
    "reasoning": "brief explanation"
}}

Rules:
- Choose exactly one category.
- Use only the information present in the document and extracted fields.
- Do not invent information.
- Confidence must be between 0.0 and 1.0.
- Keep reasoning brief.
"""

    llm_result = call_llm(prompt)
    result = llm_result.text

    return ClassificationOutput.model_validate_json(result)