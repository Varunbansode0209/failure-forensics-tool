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
from app.tracing.decorator import start_trace, end_trace
from app.storage.json_store import save_trace

def run_pipeline(document_text: str):
    start_trace()

    try:
        intake_result = intake(IntakeInput(document_text=document_text))

        extraction_result = extraction(
            ExtractionInput(
                document_text=intake_result.document_text,
                document_type=intake_result.document_type,
            )
        )

        classification_result = classification(
            ClassificationInput(
                document_text=intake_result.document_text,
                extracted_fields=extraction_result.fields,
            )
        )

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

    finally:
        trace = end_trace()

        if trace:
            file_path = save_trace(trace)

            print("\n===== TRACE =====")
            print("Trace ID:", trace.trace_id)
            print("Status:", trace.status)
            print("Spans:", len(trace.spans))
            print("Saved to:", file_path)

if __name__ == "__main__":
    document = """
    ABC Pvt Ltd and XYZ Ltd agree to a 12-month software development
    contract starting on January 1, 2026. The total contract value
    is ₹6,00,000.
    """

    result = run_pipeline(document)

    print("\n===== PIPELINE RESULT =====")
    print(result)