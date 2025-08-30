# schemas.py
# --- 1. Import necessary libraries ---
# Pydantic's BaseModel is the foundation for creating data models.
# Field is used to add extra information (like descriptions) to a field.
from pydantic import BaseModel, Field
# List is a type hint from Python's standard typing library.
from typing import List

# --- 2. Define Pydantic Models for QA Documents ---

class UserStory(BaseModel):
    """
    Defines the schema for a user story.
    A user story is a brief, informal description of a feature from an end-user perspective.
    """
    title: str = Field(description="A clear and concise title for the user story.")
    description: str = Field(description="A brief description of the user story in the 'As a [user], I want to [action], so that [goal]' format.")
    acceptance_criteria: List[str] = Field(description="A list of measurable and verifiable conditions that must be met to consider the user story complete.")

class TestCase(BaseModel):
    """
    Defines the schema for a single test case.
    A test case is a set of conditions or variables under which a tester will determine if a system is working correctly.
    """
    test_case_id: str = Field(description="A unique identifier for the test case.")
    description: str = Field(description="A clear description of what is being tested.")
    test_type: str = Field(description="The type of test case, e.g., 'Positive' or 'Negative'. Positive tests verify that the system behaves as expected with valid inputs, while negative tests ensure the system handles invalid inputs gracefully.")
    steps: List[str] = Field(description="A list of step-by-step instructions to execute the test case.")
    expected_result: str = Field(description="The expected outcome of the test case.")

class TestCases(BaseModel):
    """
    Defines the schema to contain a comprehensive list of test cases,
    categorized into positive and negative types.
    """
    positive: List[TestCase] = Field(description="A list of positive test cases.")
    negative: List[TestCase] = Field(description="A list of negative test cases.")

class Requirement(BaseModel):
    """
    Defines the schema for a single requirement.
    A requirement is a functional or non-functional specification that the system must meet.
    """
    requirement_id: str = Field(description="A unique identifier for the requirement.")
    description: str = Field(description="The functional or non-functional requirement.")
    priority: str = Field(description="The priority of the requirement, e.g., 'High', 'Medium', 'Low'.")

# This is a key part of the solution. LangChain's `with_structured_output` requires
# a single Pydantic model. We use this container class to correctly handle a list of
# Requirement objects as a single, valid output schema. 
class RequirementsList(BaseModel):
    """
    A container model for a list of requirements.
    This wrapper class is necessary to conform to the input requirements of LangChain's structured output.
    """
    requirements: List[Requirement] = Field(description="A list of functional and non-functional requirements.")

class RequirementMatrix(BaseModel):
    """
    Defines the schema for a single entry in a requirements traceability matrix.
    This matrix links requirements to the test cases that validate them, ensuring
    all requirements are tested.
    """
    requirement_id: str = Field(description="The unique identifier of the requirement.")
    linked_test_cases: List[str] = Field(description="A list of test case IDs that validate this requirement.")

# Similar to the RequirementsList, this is a container model to allow LangChain
# to output a list of RequirementMatrix objects as a single, structured object.
class RequirementMatrixList(BaseModel):
    """
    A container model for a list of requirements traceability matrices.
    This wrapper class is necessary to conform to the input requirements of LangChain's structured output.
    """
    requirements_matrix: List[RequirementMatrix] = Field(description="A list of requirements traceability matrix entries.")

class FullQAOutput(BaseModel):
    """
    Defines the complete, combined schema for the final QA output.
    This model contains all the individual document models as its fields,
    allowing the LLM to generate a single, comprehensive JSON object.
    """
    user_story: UserStory = Field(description="The user story for the request.")
    requirements: List[Requirement] = Field(description="A list of all requirements derived from the request.")
    test_cases: TestCases = Field(description="All test cases (positive and negative) for the request.")
    requirements_matrix: List[RequirementMatrix] = Field(description="The traceability matrix linking requirements to test cases.")