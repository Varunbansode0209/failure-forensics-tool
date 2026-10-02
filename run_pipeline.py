#FLOW 
#Raw document-Intake-ExtractionClassification-Summarization-Final result
from app.models.pipeline_models import (
    IntakeInput,
    ExtractionInput,
    ClassificationInput,
    SummarizationInput,
)

from app.pipeline.intake import intake
from app.pipeline.extraction import extraction
from app.pipeline.classification import classification
from app.pipeline.summarization import summarization


def run_pipeline(document_text: str):
    """
    Run the complete document processing pipeline.
    """

    # Step 1: Intake
    intake_result = intake(
        IntakeInput(document_text=document_text)
    )

    # Step 2: Extraction
    extraction_result = extraction(
        ExtractionInput(
            document_text=intake_result.document_text,
            document_type=intake_result.document_type,
        )
    )

    # Step 3: Classification
    classification_result = classification(
        ClassificationInput(
            document_text=intake_result.document_text,
            extracted_fields=extraction_result.fields,
        )
    )

    # Step 4: Summarization
    summarization_result = summarization(
        SummarizationInput(
            document_text=intake_result.document_text,
            document_type=intake_result.document_type,
            category=classification_result.category,
            extracted_fields=extraction_result.fields,
        )
    )

    return {
        "intake": intake_result,
        "extraction": extraction_result,
        "classification": classification_result,
        "summarization": summarization_result,
    }


if __name__ == "__main__":
    document = """
    ABC Pvt Ltd and XYZ Ltd agree to a 12-month software development
    contract starting on January 1, 2026. The total contract value
    is ₹6,00,000.
    """

    result = run_pipeline(document)

    print("\n===== PIPELINE RESULT =====")
    print(result)