# schemas.py

# Import necessary classes from pydantic for creating data models.
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, root_validator

# --- 1. Core Document Schemas ---
# These are the fundamental building blocks for our QA documents.

class UserStory(BaseModel):
    """
    Defines the structure for a user story.
    """
    title: str = Field(..., description="The title of the user story.")
    description: str = Field(..., description="The detailed description of the user story.")
    acceptance_criteria: List[str] = Field(..., description="A list of acceptance criteria for the user story.")

class Requirement(BaseModel):
    """
    Defines the structure for a functional or non-functional requirement.
    Includes a validator to ensure the 'requirement_id' has a prefix.
    """
    requirement_id: str = Field(..., description="A unique identifier for the requirement (e.g., FR-01, NFR-01).")
    description: str = Field(..., description="The detailed description of the requirement.")
    
    @root_validator(pre=True)
    def add_prefix_if_missing(cls, values):
        """
        Validates the data before Pydantic model creation.
        If the 'requirement_id' is just a number (e.g., '1'), it adds a
        'FR-' prefix to it. This handles cases where the LLM omits the prefix.
        """
        req_id = values.get('requirement_id')
        if req_id and req_id.isdigit():
            values['requirement_id'] = f"FR-{int(req_id):02}"
        return values

class TestCase(BaseModel):
    """
    Defines the structure for a test case, including steps and expected results.
    Includes a validator to ensure the 'test_case_id' has a prefix.
    """
    test_case_id: str = Field(..., description="A unique identifier for the test case (e.g., TC-01).")
    description: str = Field(..., description="The detailed description of the test case.")
    # The 'Literal' type ensures that 'test_type' can only be 'positive' or 'negative'.
    test_type: Literal['positive', 'negative'] = Field(..., description="The type of test case, either 'positive' or 'negative'.")
    steps: List[str] = Field(..., description="A list of steps to execute the test case.")
    expected_result: str = Field(..., description="The expected outcome of the test case.")

    @root_validator(pre=True)
    def add_prefix_to_test_case_id(cls, values):
        """
        Adds a 'TC-' prefix to the test_case_id if it's just a number.
        This handles cases where the LLM omits the prefix.
        """
        tc_id = values.get('test_case_id')
        if tc_id and tc_id.isdigit():
            values['test_case_id'] = f"TC-{int(tc_id):02}"
        return values

class RequirementMatrix(BaseModel):
    """
    Defines a single entry in the requirements traceability matrix.
    """
    requirement_id: str = Field(..., description="The ID of the requirement from the Requirements list.")
    linked_test_cases: List[str] = Field(..., description="A list of test case IDs linked to the requirement.")

# --- 2. List and Composite Schemas ---
# These schemas are used to wrap the core schemas into lists or a complete output structure.

class UserStoriesList(BaseModel):
    """
    A container for a list of user stories.
    """
    user_stories: List[UserStory] = Field(..., description="A list of user stories.")

class RequirementsList(BaseModel):
    """
    A container for a list of requirements.
    """
    requirements: List[Requirement] = Field(..., description="A list of requirements.")

# Re-defining TestCases to be a simple dictionary container for the lists.
# This structure helps in organizing positive and negative test cases.
class TestCases(BaseModel):
    """
    A container for separate lists of positive and negative test cases.
    """
    positive: List[TestCase] = Field(..., description="A list of positive test cases.")
    negative: List[TestCase] = Field(..., description="A list of negative test cases.")


class RequirementMatrixList(BaseModel):
    """
    A container for a list of requirement traceability matrix entries.
    """
    requirements_matrix: List[RequirementMatrix] = Field(..., description="A list of requirement traceability matrix entries.")

class FullQAOutput(BaseModel):
    """
    The complete schema for the 'All Documents' output,
    containing all other schemas.
    """
    user_story: UserStory = Field(..., description="The generated user story.")
    requirements: RequirementsList = Field(..., description="The generated list of requirements.")
    test_cases: TestCases = Field(..., description="The generated list of test cases.")
    requirements_matrix: RequirementMatrixList = Field(..., description="The generated requirements traceability matrix.")