"""Request bodies with their limits."""

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def _length(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("the question is empty")
        if len(value) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"the question is {len(value)} characters long; the limit is {MAX_QUESTION_CHARS}"
            )
        return value


class VerifyRequest(AskRequest):
    answer: str = Field(min_length=1, max_length=4000)
    passage_ids: list[str] = Field(min_length=1, max_length=10)
