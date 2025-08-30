# main.py

# --- 1. Import Libraries ---
# Standard library imports
import os
import json
from io import BytesIO
from typing import List

# Third-party library imports
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# LangChain library imports for LLM integration
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Pydantic schemas from our local file for structured output
from schemas import (
    FullQAOutput,
    UserStory,
    Requirement,
    TestCases,
    RequirementMatrix,
    RequirementsList,
    RequirementMatrixList,
)
# Database functions from our local file to save and retrieve data
from db import save_prompt_run, get_all_runs, delete_prompt_runs


# --- 2. Configuration and Setup ---

# Load environment variables (e.g., API keys) from a .env file
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if API keys are configured correctly and stop the app if they are missing
if not gemini_api_key:
    st.error("Google API key not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()
if not groq_api_key:
    st.error("Groq API key not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()

# Load and apply custom CSS for styling the Streamlit app
try:
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.error("The 'styles/style.css' file was not found. Please create it.")
    st.stop()

# Set the page configuration for the Streamlit app
st.set_page_config(
    layout="wide",  # Use a wide layout to maximize screen real estate
    page_title="QA Agent",
    page_icon="🤖",
    initial_sidebar_state="auto"
)


# --- 3. Define Prompt Templates ---

# The prompt for the all-in-one generation. It guides the LLM to output a single JSON object
# that includes all document types.
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent that creates detailed documentation from a single user request. "
                 "You must provide a User Story, Requirements, Test Cases (positive and negative), and a Requirements Traceability Matrix. "
                 "Ensure the final output is a single, valid JSON object that strictly conforms to the specified schema."),
    ("human", "User Request: {input}"),
])

# Prompts for individual document generation. Each is tailored to a specific task
# and Pydantic schema to improve accuracy and reduce hallucination.
user_story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate a user story based on the user's request, "
                 "ensuring the output is a single, valid JSON object that strictly conforms to the UserStory schema."),
    ("human", "User Request: {input}"),
])
requirements_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate functional and non-functional requirements from the user's request, "
                 "ensuring the output is a single, valid JSON object that strictly conforms to the Requirements schema."),
    ("human", "User Request: {input}"),
])
test_cases_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate a comprehensive list of positive and negative test cases from the user's request, "
                 "ensuring the output is a single, valid JSON object that strictly conforms to the TestCases schema."),
    ("human", "User Request: {input}"),
])
req_matrix_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate a requirements traceability matrix by linking requirements and test cases "
                 "from the user's request, ensuring the output is a single, valid JSON object that strictly conforms to the RequirementsMatrix schema."),
    ("human", "User Request: {input}"),
])


# --- 4. Helper Functions for Data Manipulation and Downloads ---

def clear_text():
    """Clears the user input text area and the generated output from session state.
    This function is called by the 'Clear Text' button."""
    st.session_state.user_input = ""
    if 'qa_docs_dict' in st.session_state:
        del st.session_state.qa_docs_dict

def to_excel(dfs):
    """
    Converts a dictionary of pandas DataFrames into a single, multi-sheet Excel file.
    
    Args:
        dfs (dict): A dictionary where keys are sheet names and values are DataFrames.
    
    Returns:
        bytes: The content of the Excel file as a byte stream.
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    for sheet_name, df in dfs.items():
        # Drop the temporary "Count" column before saving to Excel for a cleaner file.
        df_to_save = df.drop(columns=['Count'], errors='ignore')
        df_to_save.to_excel(writer, index=False, sheet_name=sheet_name)
    writer.close()
    return output.getvalue()

def to_excel_history(runs):
    """
    Converts a list of historical prompt run objects into a multi-sheet Excel file.
    It parses the JSON output from each run and organizes it into DataFrames.
    
    Args:
        runs (list): A list of PromptRun objects from the database.
        
    Returns:
        bytes: The content of the Excel file as a byte stream.
    """
    user_stories_data = []
    requirements_data = []
    test_cases_data = []
    reqs_matrix_data = []

    for run in runs:
        try:
            output_data = json.loads(run.output_json)
            # Process and flatten the JSON data into a list of dictionaries for each document type.
            us = output_data.get('user_story', {})
            user_stories_data.append({
                'Timestamp': run.timestamp,
                'Input_Prompt': run.input_prompt,
                'User_Story_Title': us.get('title'),
                'User_Story_Description': us.get('description'),
                'Acceptance_Criteria': "\n".join(us.get('acceptance_criteria', []))
            })
            for req in output_data.get('requirements', []):
                requirements_data.append({
                    'Timestamp': run.timestamp,
                    'Input_Prompt': run.input_prompt,
                    'Requirement_ID': req.get('requirement_id'),
                    'Requirement_Description': req.get('description')
                })
            all_test_cases = output_data.get('test_cases', {}).get('positive', []) + output_data.get('test_cases', {}).get('negative', [])
            for tc in all_test_cases:
                test_cases_data.append({
                    'Timestamp': run.timestamp,
                    'Input_Prompt': run.input_prompt,
                    'Test_Case_ID': tc.get('test_case_id'),
                    'Test_Case_Type': tc.get('type'),
                    'Test_Case_Description': tc.get('description'),
                    'Steps': "\n".join(tc.get('steps', [])),
                    'Expected_Result': tc.get('expected_result')
                })
            for rm in output_data.get('requirements_matrix', []):
                reqs_matrix_data.append({
                    'Timestamp': run.timestamp,
                    'Input_Prompt': run.input_prompt,
                    'Requirement_ID': rm.get('requirement_id'),
                    'Linked_Test_Cases': ", ".join(rm.get('linked_test_cases', []))
                })
        except json.JSONDecodeError:
            st.warning(f"Skipping malformed history entry from {run.timestamp}.")

    # Create a dictionary of DataFrames to be passed to the Excel writer
    dfs_to_write = {
        'User Stories': pd.DataFrame(user_stories_data),
        'Requirements': pd.DataFrame(requirements_data),
        'Test Cases': pd.DataFrame(test_cases_data),
        'Reqs Matrix': pd.DataFrame(reqs_matrix_data)
    }

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    for sheet_name, df in dfs_to_write.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()
    return output.getvalue()


# --- 5. Main Streamlit App UI and Logic ---

# Create a two-column layout for the title and tab selection
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🤖 Automated QA Agent")
with col2:
    selected_tab = st.selectbox(
        "Select Tab",
        ["Generate New Docs", "Download & Manage History"],
        label_visibility="hidden"
    )
st.divider()

# --- Content for the "Generate New Docs" Tab ---
if selected_tab == "Generate New Docs":
    st.header("Generate New Docs")
    
    # UI selectors for LLM model, output format, and content type
    model_choice = st.selectbox(
        "Select LLM Model",
        ("Gemini", "Groq"),
        key="model_selectbox"
    )
    if model_choice == "Gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
    elif model_choice == "Groq":
        llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)
    
    output_format = st.selectbox(
        "Select Output Format",
        ("JSON", "Table"),
        key="format_selectbox"
    )

    output_content = st.selectbox(
        "Select Output Content",
        ["All Documents", "User Story", "Requirements", "Test Cases", "Requirements Matrix"],
        key="output_content_selectbox"
    )
    st.markdown("---")

    # The main user input text area
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_area("Your Request", height=150, key="user_input")
    
    # Columns for the control buttons
    col_main_1, col_main_2, col_main_3 = st.columns([1, 1, 1])

    with col_main_1:
        # The main "Generate" button logic
        if st.button("✨ Generate QA Docs", key="generate_button"):
            if not user_input:
                st.warning("Please enter a request.")
            else:
                with st.spinner(f"Generating {output_content} using {model_choice}... This may take a moment."):
                    try:
                        qa_docs_dict = {}
                        
                        # Use conditional logic to run the correct LangChain expression based on user selection
                        if output_content == "All Documents":
                            qa_agent_chain = base_prompt | llm.with_structured_output(FullQAOutput)
                            qa_docs = qa_agent_chain.invoke({"input": user_input})
                            qa_docs_dict = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "User Story":
                            user_story_chain = user_story_prompt | llm.with_structured_output(UserStory)
                            qa_docs = user_story_chain.invoke({"input": user_input})
                            qa_docs_dict['user_story'] = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "Requirements":
                            # Use the RequirementsList container class to get a list of requirements
                            requirements_chain = requirements_prompt | llm.with_structured_output(RequirementsList)
                            qa_docs = requirements_chain.invoke({"input": user_input})
                            # Extract the list of requirements from the container model and dump to dictionary
                            qa_docs_dict['requirements'] = [item.model_dump() for item in qa_docs.requirements]
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "Test Cases":
                            test_cases_chain = test_cases_prompt | llm.with_structured_output(TestCases)
                            qa_docs = test_cases_chain.invoke({"input": user_input})
                            qa_docs_dict['test_cases'] = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "Requirements Matrix":
                            # Use the RequirementMatrixList container class for the matrix list
                            req_matrix_chain = req_matrix_prompt | llm.with_structured_output(RequirementMatrixList)
                            qa_docs = req_matrix_chain.invoke({"input": user_input})
                            # Extract the list from the container model and dump to dictionary
                            qa_docs_dict['requirements_matrix'] = [item.model_dump() for item in qa_docs.requirements_matrix]
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        # Store the generated data in Streamlit's session state for display
                        st.session_state.qa_docs_dict = qa_docs_dict
                        st.success("✅ Documentation Generated and Saved!")
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    
    with col_main_2:
        st.button("❌ Clear Text", on_click=clear_text, key="clear_button")

    # --- Section for displaying the generated output ---
    if 'qa_docs_dict' in st.session_state:
        qa_docs_dict = st.session_state.qa_docs_dict
        st.divider()

        # Display the output in JSON or table format based on user's selection
        if output_format == "JSON":
            st.markdown("### <font color='#4CAF50'>Generated Output (JSON)</font>", unsafe_allow_html=True)
            st.json(qa_docs_dict)
        else:
            st.markdown("### <font color='#4CAF50'>Generated Output (Table)</font>", unsafe_allow_html=True)
            
            dfs_to_download = {}
            
            # Conditionally display each table if its data exists in the dictionary
            if 'user_story' in qa_docs_dict:
                us_dict = qa_docs_dict['user_story']
                us_dict['acceptance_criteria'] = "\n".join(us_dict['acceptance_criteria'])
                user_story_df = pd.DataFrame([us_dict])
                user_story_df.insert(0, "Count", range(1, 1 + len(user_story_df)))
                dfs_to_download['User Story'] = user_story_df
                st.subheader("User Story")
                st.dataframe(user_story_df, width='stretch', hide_index=True, column_config={
                    "User_Story_Description": st.column_config.Column(width="large"),
                    "Acceptance_Criteria": st.column_config.Column(width="large")
                })
            
            if 'requirements' in qa_docs_dict:
                reqs_list = qa_docs_dict['requirements']
                reqs_df = pd.DataFrame(reqs_list)
                reqs_df.insert(0, "Count", range(1, 1 + len(reqs_df)))
                dfs_to_download['Requirements'] = reqs_df
                st.subheader("Requirements")
                st.dataframe(reqs_df, width='stretch', hide_index=True, column_config={
                    "Requirement_Description": st.column_config.Column(width="large")
                })
            
            if 'test_cases' in qa_docs_dict:
                test_cases_list = qa_docs_dict['test_cases']['positive'] + qa_docs_dict['test_cases']['negative']
                for tc in test_cases_list:
                    tc['steps'] = "\n".join(tc['steps'])
                test_cases_df = pd.DataFrame(test_cases_list)
                test_cases_df.insert(0, "Count", range(1, 1 + len(test_cases_df)))
                dfs_to_download['Test Cases'] = test_cases_df
                st.subheader("Test Cases")
                st.dataframe(test_cases_df, width='stretch', hide_index=True, column_config={
                    "Test_Case_Description": st.column_config.Column(width="large"),
                    "Steps": st.column_config.Column(width="large"),
                    "Expected_Result": st.column_config.Column(width="large")
                })
            
            if 'requirements_matrix' in qa_docs_dict:
                req_matrix_list = qa_docs_dict['requirements_matrix']
                for rm in req_matrix_list:
                    rm['linked_test_cases'] = ", ".join(rm['linked_test_cases'])
                req_matrix_df = pd.DataFrame(req_matrix_list)
                req_matrix_df.insert(0, "Count", range(1, 1 + len(req_matrix_df)))
                dfs_to_download['Reqs Matrix'] = req_matrix_df
                st.subheader("Requirements Matrix")
                st.dataframe(req_matrix_df, width='stretch', hide_index=True, column_config={
                    "Linked_Test_Cases": st.column_config.Column(width="large")
                })
            
            # Download buttons for JSON and Excel files
            with col_main_3:
                st.markdown("### **Download Files**")
                json_str = json.dumps(qa_docs_dict, indent=2)
                st.download_button(
                    label="Download as JSON",
                    data=json_str,
                    file_name="qa_docs.json",
                    mime="application/json",
                    key="download_json"
                )
                excel_data = to_excel(dfs_to_download)
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="qa_docs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )

# --- Content for the "Download & Manage History" Tab ---
elif selected_tab == "Download & Manage History":
    st.header("Download & Manage Past Runs")
    runs = get_all_runs()
    
    if not runs:
        st.info("No past runs found.")
    else:
        # State management for selected runs to be downloaded or deleted
        if 'selected_run_ids' not in st.session_state:
            st.session_state.selected_run_ids = []

        def on_checkbox_change(run_id, is_checked):
            """Callback function for the checkboxes to manage selected runs."""
            if is_checked and run_id not in st.session_state.selected_run_ids:
                st.session_state.selected_run_ids.append(run_id)
            elif not is_checked and run_id in st.session_state.selected_run_ids:
                st.session_state.selected_run_ids.remove(run_id)

        # Display a checkbox for each historical run
        for run in runs:
            checkbox_label = f"Select: Run on {run.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {run.input_prompt[:50]}..."
            is_checked = run.id in st.session_state.selected_run_ids
            st.checkbox(checkbox_label, value=is_checked, on_change=on_checkbox_change, args=(run.id, not is_checked), key=f"checkbox_{run.id}")
        
        st.divider()

        # Control buttons for history management
        col_buttons_1, col_buttons_2, col_buttons_3 = st.columns([1, 1, 1])

        with col_buttons_1:
            if st.session_state.selected_run_ids:
                selected_runs = [run for run in runs if run.id in st.session_state.selected_run_ids]
                excel_data_selected = to_excel_history(selected_runs)
                st.download_button(
                    label="Download Selected",
                    data=excel_data_selected,
                    file_name="qa_selected_history.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_selected"
                )
            else:
                st.info("Select items to download.")

        with col_buttons_2:
            excel_data_all = to_excel_history(runs)
            st.download_button(
                label="Download All",
                data=excel_data_all,
                file_name="qa_all_history.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_all"
            )

        with col_buttons_3:
            if st.button("Delete Selected", key="delete_selected"):
                if st.session_state.selected_run_ids:
                    delete_prompt_runs(st.session_state.selected_run_ids)
                    st.session_state.selected_run_ids = []
                    st.success("Selected items deleted successfully!")
                    st.rerun()  # Rerun the app to update the display after deletion

        st.divider()

        # Display details of selected items in an expandable format
        if st.session_state.selected_run_ids:
            st.subheader("Details of Selected Items")
            for run_id in st.session_state.selected_run_ids:
                run = next((r for r in runs if r.id == run_id), None)
                if run:
                    with st.expander(f"Run on {run.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                        st.subheader("Original Prompt")
                        st.text(run.input_prompt)
                        st.subheader("Generated Output")
                        st.json(json.loads(run.output_json))