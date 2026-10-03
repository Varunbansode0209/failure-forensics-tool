#Intake is the first pipeline stage.
#It receives raw document text, asks the LLM to identify the document type and language, and returns a validated IntakeOutput

#THE FLOW IS 
# RAW TEXT - INTAKE INPUT - OPENAI - JSON RESPONSE - PYDANTIC VALIDATION - INTAKE OUTPUT



from dotenv import load_dotenv



from app.models.pipeline_models import IntakeInput,IntakeOutput
from app.tracing.decorator import traced
from app.llm.client import call_llm


@traced
def intake(document : IntakeInput) -> IntakeOutput:
    """
    Identify the document type and language from raw document text
    """

    prompt = f"""
    you are the intake stage of an AI document processing pipeline.

    Analyze the document below.

    Your tasks:
    1.Identify the document type.
    2.Identify the language of the document.
    3.Provide a confidence score between 0.0 and 1.0.


    Allowed document types:
    -contract
    -invoice
    -report
    -correspondence 

    Return ONLY valid JSON using this structure:
    {{
            "document_text": "original document text",
            "document_type": "contract | invoice | report | correspondence",
            "language": "language name",
            "confidence": 0.0   
    }}

        Rules:
        - Do not invent information.
        - Use the closest allowed document type.
        - Confidence must be between 0.0 and 1.0.
        - Preserve the original document text exactly.

    Document:
    {document.document_text}
    """

    llm_result = call_llm(prompt)
    result = llm_result.text

    

    

    return IntakeOutput.model_validate_json(result)