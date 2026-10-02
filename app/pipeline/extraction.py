#The purpose of extraction.py is to take the raw document + document type identified by Intake and convert the document into structured fields.

#FLOW
# INTAKE - type(document) - extract - gemini 3.8flash --extract relevant fields -- structured JSON - pydantic validation - extraction output 

import os 

from dotenv import load_dotenv
from google import genai

from app.models.pipeline_models import ExtractionInput,ExtractionOutput

load_dotenv()

client= genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extraction(document: ExtractionInput) -> ExtractionOutput:
    """
    Extract structured fields from a document.
    """

    prompt = f"""
    You are the extraction stage of an AI document processing pipeline.

    Extract the important fields from the document.

    Document type:
    {document.document_type}

    Document:
    {document.document_text}

Return ONLY valid JSON using exactly this structure:

{{
    "document_type": "{document.document_type}",
    "fields": {{
        "field_name": "field_value"
    }},
    "confidence": 0.0
}}

Rules:
- Extract only information explicitly present in the document.
- Do not invent missing values.
- Use null when a field is not present.
- Extract fields relevant to the document type.
- Confidence must be between 0.0 and 1.0.
"""


    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    result = interaction.output_text

    return ExtractionOutput.model_validate_json(result)