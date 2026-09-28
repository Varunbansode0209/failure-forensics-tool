from pydantic import BaseModel, Field
from typing import Optional,List,Dict



#STEP 1:- INTAKE

class IntakeInput(BaseModel):
    document_text:str = Field(
        min_length=1,
        description="Raw document text provided to the pipeline"
    )
    

class IntakeOutput(BaseModel):
    document_type:str
    document_text:str
    language:str
    confidence:float = Field(ge=0.0, le=1.0)


#STEP 2:- EXTRACTION

class ExtractionInput(BaseModel):
    document_text: str
    document_type:str


class ExtractionOutput(BaseModel):
    document_type:str
    fields: Dict[str, Optional[str]]
    confidence: float = Field(ge=0.0 , le=1.0)



#STEP 3 :- CLASSIFICATION 

class ClassificationInput(BaseModel):
    document_text: str
    extracted_fields: Dict[str, Optional[str]]


class ClassificationOutput(BaseModel):
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


#STEP 4:- SUMMARIZATION

class SummarizationInput(BaseModel):
    document_text: str
    document_type: str
    category: str
    extracted_fields: Dict[str, Optional[str]]


class SummarizationOutput(BaseModel):
    summary: str
    key_points: List[str]
    confidence: float = Field(ge=0.0, le=1.0)