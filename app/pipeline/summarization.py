#The Summarization step takes everything the previous stages have understood about the document and produces a human-readable summary.

#FLOW
# DOCUMENT - INTAKE --document_type-- - EXTRACTION --extraction_field-- - CLASSIFICATION --category-- - SUMMARIZATION - GEMINI 3.8FLASH - STRUCTURE JSON - PYDANTIC VALIDATION - SUMMARIZATIONOUTPUT


import os 

from dotenv import load_dotenv
from google import genai

from app.models.pipeline_models import(
    SummarizationInput,
    SummarizationOutput,
)
from app.tracing.decorator import traced

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    result = interaction.output_text

    return SummarizationOutput.model_validate_json(result)