from pydantic import BaseModel, Field
from typing import List
from chat_agent.llm.structured import parse_structured

class ExtractedLead(BaseModel):
    first_name: str = Field(description="Vorname")
    last_name: str = Field(description="Nachname")
    company: str = Field(description="Unternehmen oder 'Freiberuflich' falls unbekannt")
    confidence_score: float = Field(description="0..1")

class LeadList(BaseModel):
    leads: List[ExtractedLead]

raw_text = """
Ich habe mich gestern mit Sarah Connor von Cyberdyne Systems getroffen.
Außerdem hat Kyle Reese angerufen; er ist momentan ohne Job.
"""

result = parse_structured(LeadList, raw_text)
print(result)
print(result.leads[0].first_name)
