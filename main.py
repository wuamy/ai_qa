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
from pydantic import BaseModel, Field

# LangChain library imports for LLM integration and structured output
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Pydantic schemas from our local file for structured output
from schemas import (
    FullQAOutput,
    UserStory,
    UserStoriesList,
    Requirement,
    TestCases,
    RequirementMatrix,
    RequirementsList,
    RequirementMatrixList,
)
# Database functions from our local file to save and retrieve data
from db import save_prompt_run, get_all_runs, delete_prompt_runs


# --- 2. Configuration and Setup ---

# Load environment variables (e.g., API keys) from a .env file.
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if API keys are configured correctly and stop the app if they are missing.
if not gemini_api_key:
    st.error("Google API key not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()
if not groq_api_key:
    st.error("Groq API key not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()

# Load and apply custom CSS for styling the Streamlit app.
try:
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.error("The 'styles/style.css' file was not found. Please create it.")
    st.stop()

# Set the page configuration for the Streamlit app.
st.set_page_config(
    layout="wide",
    page_title="QA Agent",
    page_icon="🤖",
    initial_sidebar_state="auto"
)


# --- 3. Define Prompt Templates ---

base_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent that creates detailed documentation from a single user request. "
                 "You must provide a User Story, Requirements, Test Cases (positive and negative), and a Requirements Traceability Matrix. "
                 "Ensure the final output is a single, valid JSON object that strictly conforms to the specified schema."),
    ("human", "User Request: {input}"),
])

user_story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate a user story based on the user's request. "
                 "Generate any number of user stories as requested by the user, and ensure the output is a single, "
                 "valid JSON object that strictly conforms to the UserStoriesList schema."),
    ("human", "User Request: {input}"),
])
requirements_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate functional and non-functional requirements from the user's request, "
                 "ensuring the output is a single, valid JSON object that strictly conforms to the RequirementsList schema."),
    ("human", "User Request: {input}"),
])
test_cases_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate a comprehensive list of positive and negative test cases from the user's request, "
                 "ensuring the output is a single, valid JSON object that strictly conforms to the TestCases schema. "
                 "You must generate **at least one positive and one negative test case** to populate their respective lists in the output, even if the user's request is brief."),
    ("human", "User Request: {input}"),
])
req_matrix_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a specialized QA agent. Your task is to generate a requirements traceability matrix by linking requirements and test cases "
                 "from the user's request, ensuring the output is a single, valid JSON object that strictly conforms to the RequirementMatrixList schema."),
    ("human", "User Request: {input}"),
])


# --- 4. Helper Functions for Data Manipulation and Downloads ---

def clear_text():
    """Clears the user input text area and the generated output from session state."""
    st.session_state.user_input = ""
    if 'qa_docs_dict' in st.session_state:
        del st.session_state.qa_docs_dict

def to_excel(dfs):
    """
    Converts a dictionary of pandas DataFrames into a single, multi-sheet Excel file.
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    for sheet_name, df in dfs.items():
        df_to_save = df.drop(columns=['Count'], errors='ignore')
        df_to_save.to_excel(writer, index=False, sheet_name=sheet_name)
    writer.close()
    return output.getvalue()

def to_excel_history(runs):
    """
    Converts a list of historical prompt run objects into a multi-sheet Excel file.
    """
    user_stories_data = []
    requirements_data = []
    test_cases_data = []
    reqs_matrix_data = []

    for run in runs:
        try:
            output_data = json.loads(run.output_json)
            user_story_list = []
            if 'user_story' in output_data:
                us_data = output_data.get('user_story', {})
                if isinstance(us_data, dict):
                    user_story_list.append(us_data)
            elif 'user_stories' in output_data and isinstance(output_data['user_stories'], dict):
                user_story_list.extend(output_data['user_stories'].get('user_stories', []))
            
            for us in user_story_list:
                user_stories_data.append({
                    'Timestamp': run.timestamp,
                    'Input_Prompt': run.input_prompt,
                    'User_Story_Title': us.get('title'),
                    'User_Story_Description': us.get('description'),
                    'Acceptance_Criteria': "\n".join(us.get('acceptance_criteria', []))
                })

            requirements_list = []
            if 'requirements' in output_data:
                req_data = output_data.get('requirements', {})
                if isinstance(req_data, list):
                    requirements_list.extend(req_data)
                elif isinstance(req_data, dict):
                    requirements_list.extend(req_data.get('requirements', []))

            for req in requirements_list:
                requirements_data.append({
                    'Timestamp': run.timestamp,
                    'Input_Prompt': run.input_prompt,
                    'Requirement_ID': req.get('requirement_id'),
                    'Requirement_Description': req.get('description')
                })

            all_test_cases = []
            if 'test_cases' in output_data and isinstance(output_data['test_cases'], dict):
                all_test_cases = output_data['test_cases'].get('positive', []) + output_data['test_cases'].get('negative', [])
            
            for tc in all_test_cases:
                test_cases_data.append({
                    'Timestamp': run.timestamp,
                    'Input_Prompt': run.input_prompt,
                    'Test_Case_ID': tc.get('test_case_id'),
                    'Test_Case_Type': tc.get('test_type'),
                    'Test_Case_Description': tc.get('description'),
                    'Steps': "\n".join(tc.get('steps', [])),
                    'Expected_Result': tc.get('expected_result')
                })
            
            requirements_matrix_list = []
            if 'requirements_matrix' in output_data:
                req_matrix_data = output_data.get('requirements_matrix', {})
                if isinstance(req_matrix_data, list):
                    requirements_matrix_list.extend(req_matrix_data)
                elif isinstance(req_matrix_data, dict):
                    requirements_matrix_list.extend(req_matrix_data.get('requirements_matrix', []))

            for rm in requirements_matrix_list:
                reqs_matrix_data.append({
                    'Timestamp': run.timestamp,
                    'Input_Prompt': run.input_prompt,
                    'Requirement_ID': rm.get('requirement_id'),
                    'Linked_Test_Cases': ", ".join(rm.get('linked_test_cases', []))
                })
        except json.JSONDecodeError:
            st.warning(f"Skipping malformed history entry from {run.timestamp}.")

    dfs_to_write = {
        'User Stories': pd.DataFrame(user_stories_data),
        'Requirements': pd.DataFrame(requirements_data),
        'Test Cases': pd.DataFrame(test_cases_data),
        'Reqs Matrix': pd.DataFrame(reqs_matrix_data)
    }

    output = Bytesio()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    for sheet_name, df in dfs_to_write.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()
    return output.getvalue()


# --- 5. Main Streamlit App UI and Logic ---

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
    col_header_left, col_header_right = st.columns([0.7, 0.3])
    
    with col_header_left:
        st.header("Generate New Docs")
    
    with col_header_right:
        if 'qa_docs_dict' in st.session_state and st.session_state.qa_docs_dict:
            st.markdown("### **Download Files**")
            dfs_to_download = {}
            qa_docs_dict = st.session_state.qa_docs_dict
            
            if 'user_stories' in qa_docs_dict:
                us_data = qa_docs_dict.get('user_stories', {})
                us_list = us_data.get('user_stories', []) if isinstance(us_data, dict) else us_data
                df_us = pd.DataFrame(us_list)
                if not df_us.empty:
                    dfs_to_download['User Stories'] = df_us
            
            if 'requirements' in qa_docs_dict:
                reqs_data = qa_docs_dict.get('requirements', {})
                reqs_list = reqs_data.get('requirements', []) if isinstance(reqs_data, dict) else reqs_data
                df_reqs = pd.DataFrame(reqs_list)
                if not df_reqs.empty:
                    dfs_to_download['Requirements'] = df_reqs

            if 'test_cases' in qa_docs_dict:
                tc_data = qa_docs_dict.get('test_cases', {})
                tc_list = tc_data.get('positive', []) + tc_data.get('negative', [])
                df_tc = pd.DataFrame(tc_list)
                if not df_tc.empty:
                    dfs_to_download['Test Cases'] = df_tc
            
            if 'requirements_matrix' in qa_docs_dict:
                rm_data = qa_docs_dict.get('requirements_matrix', {})
                rm_list = rm_data.get('requirements_matrix', []) if isinstance(rm_data, dict) else rm_data
                df_rm = pd.DataFrame(rm_list)
                if not df_rm.empty:
                    dfs_to_download['Reqs Matrix'] = df_rm

            json_str = json.dumps(qa_docs_dict, indent=2)
            excel_data = to_excel(dfs_to_download)

            # --- Start: Side-by-Side Button Implementation ---
            col_json, col_excel = st.columns([1, 1])

            with col_json:
                st.download_button(
                    label="Download as JSON",
                    data=json_str,
                    file_name="qa_docs.json",
                    mime="application/json",
                    key="download_json"
                )

            with col_excel:
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="qa_docs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            # --- End: Side-by-Side Button Implementation ---

    st.markdown("---")
    
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

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_area("Your Request", height=150, key="user_input")
    
    col_main_1, col_main_2, _ = st.columns([1, 1, 1])

    with col_main_1:
        if st.button("✨ Generate QA Docs", key="generate_button"):
            if not user_input:
                st.warning("Please enter a request.")
            else:
                with st.spinner(f"Generating {output_content} using {model_choice}... This may take a moment."):
                    try:
                        qa_docs_dict = {}
                        
                        if output_content == "All Documents":
                            qa_agent_chain = base_prompt | llm.with_structured_output(FullQAOutput)
                            qa_docs = qa_agent_chain.invoke({"input": user_input})
                            qa_docs_dict = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "User Story":
                            user_story_chain = user_story_prompt | llm.with_structured_output(UserStoriesList)
                            qa_docs = user_story_chain.invoke({"input": user_input})
                            qa_docs_dict['user_stories'] = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "Requirements":
                            requirements_chain = requirements_prompt | llm.with_structured_output(RequirementsList)
                            qa_docs = requirements_chain.invoke({"input": user_input})
                            qa_docs_dict['requirements'] = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "Test Cases":
                            test_cases_chain = test_cases_prompt | llm.with_structured_output(TestCases)
                            qa_docs = test_cases_chain.invoke({"input": user_input})
                            qa_docs_dict['test_cases'] = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        elif output_content == "Requirements Matrix":
                            req_matrix_chain = req_matrix_prompt | llm.with_structured_output(RequirementMatrixList)
                            qa_docs = req_matrix_chain.invoke({"input": user_input})
                            qa_docs_dict['requirements_matrix'] = qa_docs.model_dump()
                            output_json_str = json.dumps(qa_docs_dict, indent=2)
                            save_prompt_run(user_input, output_json_str)

                        st.session_state.qa_docs_dict = qa_docs_dict
                        st.success("✅ Documentation Generated and Saved!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    
    with col_main_2:
        st.button("❌ Clear Text", on_click=clear_text, key="clear_button")

    if 'qa_docs_dict' in st.session_state:
        qa_docs_dict = st.session_state.qa_docs_dict
        st.divider()

        if output_format == "JSON":
            st.markdown("### <font color='#4CAF50'>Generated Output (JSON)</font>", unsafe_allow_html=True)
            st.json(qa_docs_dict)
        else:
            st.markdown("### <font color='#4CAF50'>Generated Output (Table)</font>", unsafe_allow_html=True)
            
            dfs_to_download = {}
            
            if 'user_stories' in qa_docs_dict:
                us_data = qa_docs_dict.get('user_stories', {})
                if isinstance(us_data, list):
                    us_list = us_data
                else:
                    us_list = us_data.get('user_stories', [])
                for us_dict in us_list:
                    if 'acceptance_criteria' in us_dict and isinstance(us_dict['acceptance_criteria'], list):
                        us_dict['acceptance_criteria'] = "\n".join(us_dict['acceptance_criteria'])
                user_story_df = pd.DataFrame(us_list)
                user_story_df.insert(0, "Count", range(1, 1 + len(user_story_df)))
                dfs_to_download['User Stories'] = user_story_df
                st.subheader("User Stories")
                st.dataframe(user_story_df, width='stretch', hide_index=True, column_config={
                    "User_Story_Title": st.column_config.Column(width="small"),
                    "User_Story_Description": st.column_config.Column(width="large"),
                    "Acceptance_Criteria": st.column_config.Column(width="large")
                })
            
            if 'requirements' in qa_docs_dict:
                reqs_data = qa_docs_dict.get('requirements', {})
                if isinstance(reqs_data, list):
                    reqs_list = reqs_data
                else:
                    reqs_list = reqs_data.get('requirements', [])
                reqs_df = pd.DataFrame(reqs_list)
                reqs_df.insert(0, "Count", range(1, 1 + len(reqs_df)))
                dfs_to_download['Requirements'] = reqs_df
                st.subheader("Requirements")
                st.dataframe(reqs_df, width='stretch', hide_index=True, column_config={
                    "Requirement_Description": st.column_config.Column(width="large")
                })

            if 'test_cases' in qa_docs_dict:
                test_cases_data = qa_docs_dict.get('test_cases', {})
                test_cases_list = test_cases_data.get('positive', []) + test_cases_data.get('negative', [])
                for tc in test_cases_list:
                    if 'steps' in tc and isinstance(tc['steps'], list):
                        tc['steps'] = "\n".join(tc['steps'])
                    if 'test_type' not in tc:
                        tc['test_type'] = tc.pop('type', None)
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
                req_matrix_data = qa_docs_dict.get('requirements_matrix', {})
                if isinstance(req_matrix_data, list):
                    req_matrix_list = req_matrix_data
                else:
                    req_matrix_list = req_matrix_data.get('requirements_matrix', [])
                for rm in req_matrix_list:
                    if 'linked_test_cases' in rm and isinstance(rm['linked_test_cases'], list):
                        rm['linked_test_cases'] = ", ".join(rm['linked_test_cases'])
                req_matrix_df = pd.DataFrame(req_matrix_list)
                req_matrix_df.insert(0, "Count", range(1, 1 + len(req_matrix_df)))
                dfs_to_download['Reqs Matrix'] = req_matrix_df
                st.subheader("Requirements Matrix")
                st.dataframe(req_matrix_df, width='stretch', hide_index=True, column_config={
                    "Linked_Test_Cases": st.column_config.Column(width="large")
                })

# --- Content for the "Download & Manage History" Tab ---
elif selected_tab == "Download & Manage History":
    st.header("Download & Manage Past Runs")
    runs = get_all_runs()
    
    if not runs:
        st.info("No past runs found.")
    else:
        if 'selected_run_ids' not in st.session_state:
            st.session_state.selected_run_ids = []

        def on_checkbox_change(run_id, is_checked):
            if is_checked and run_id not in st.session_state.selected_run_ids:
                st.session_state.selected_run_ids.append(run_id)
            elif not is_checked and run_id in st.session_state.selected_run_ids:
                st.session_state.selected_run_ids.remove(run_id)

        for run in runs:
            checkbox_label = f"Select: Run on {run.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {run.input_prompt[:50]}..."
            is_checked = run.id in st.session_state.selected_run_ids
            st.checkbox(checkbox_label, value=is_checked, on_change=on_checkbox_change, args=(run.id, not is_checked), key=f"checkbox_{run.id}")
        
        st.divider()

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
                    st.rerun()

        st.divider()

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