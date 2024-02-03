from langchain_google_genai import GoogleGenerativeAI as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Return a list of multiples of 2')
parser.add_argument('--language', default='Python')
args = parser.parse_args()
load_dotenv()

llm = genai(model="gemini-pro")
code_prompt = PromptTemplate(
    template="Write a {language} function that will {task}",
    input_variables=['language','task']
)

tset_code_prompt = PromptTemplate(
    input_variables=['language','code'],
    template="Write a test code for the {language} function:\n{code}"
    
)

documentaion_prompt = PromptTemplate(
    input_variables=['language','code'],
    template="Write a documentaion for the {language} function:\n{code}"
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key='code'
)

tset_code_chain = LLMChain(
    llm=llm,
    prompt=tset_code_prompt,
    output_key='test_code'
)

documentaion_chain = LLMChain(
    llm=llm,
    prompt=documentaion_prompt,
    output_key='documentation'
)
chain = SequentialChain(
    chains=[code_chain, tset_code_chain, documentaion_chain],
    input_variables=['language', 'task'],
    output_variables=['code','test_code','documentation']
)
result = chain(
    {
        'language':args.language,
        'task':args.task
        
    }
)

#print(result['text'])
print('------------GENERATED CODE--------------')
print(result['code'])
print('---------------GENERATED TEST-------------')
print(result['test_code'])
print('--------------DOCUMENTATION------------')
print(result['documentation'])