import streamlit as st
from langchain_google_genai import GoogleGenerativeAI as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("Automatic Code Generation")

# Define supported programming languages
languages = ["Python", "JavaScript", "Java", "C++","R", "Ruby", "Swift"]

# Language selection dropdown
selected_language = st.selectbox("Select Programming Language", languages)

# Task input field
task = st.text_input("Enter Task Description", "Return a list of multiples of 2")

# Generate code button
if st.button("Generate Code"):
    # Initialize Google Generative AI model
    llm = genai(model="gemini-pro")

    # Define prompt templates
    code_prompt = PromptTemplate(
        template="Write a {language} function that will {task}",
        input_variables=['language', 'task']
    )

    test_code_prompt = PromptTemplate(
        input_variables=['language', 'code'],
        template="Write a test code for the {language} function:\n{code}"
    )

    api_prompt = PromptTemplate(
    input_variables=['language','code'],
    template="Write an API for the {language} function:\n{code}"
    )
    documentation_prompt = PromptTemplate(
        input_variables=['language', 'code'],
        template="Write an API documentation for the {language} function:\n{code}"
    )

    # Create LangChain instances
    code_chain = LLMChain(
        llm=llm,
        prompt=code_prompt,
        output_key='code'
    )

    test_code_chain = LLMChain(
        llm=llm,
        prompt=test_code_prompt,
        output_key='test_code'
    )

    api_chain = LLMChain(
    llm=llm,
    prompt=api_prompt,
    output_key='api'
    )

    documentation_chain = LLMChain(
        llm=llm,
        prompt=documentation_prompt,
        output_key='documentation'
    )

    # Create SequentialChain
    chain = SequentialChain(
        chains=[code_chain, test_code_chain,api_chain,documentation_chain],
        input_variables=['language', 'task'],
        output_variables=['code', 'test_code', 'api', 'documentation']
    )

    # Execute the chain
    result = chain(
        {
            'language': selected_language,
            'task': task
        }
    )

    # Display generated code, test cases, and documentation
    st.subheader("Generated Code")
    st.code(result['code'], language='python')

    st.subheader("Generated Test Cases")
    st.code(result['test_code'], language='python')

    st.subheader("Generated API")
    st.code(result['api'], language='python')

    st.subheader("Documentation")
    st.code(result['documentation'], language='markdown')
