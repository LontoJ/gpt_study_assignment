import json

import openai
import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

api_key = st.sidebar.text_input("OpenAI API 키를 입력해 주세요.", type="password")


if api_key:
    try:
        openai.api_key = api_key
        openai.Model.list()

        st.session_state.api_key_valid = True
    except Exception as e:
        st.session_state.api_key_valid = False
        st.sidebar.error("유효하지 않은 API 키입니다. 다시 입력해 주세요.")


else:
    st.sidebar.warning("OpenAI API 키를 입력해 주세요.")


if st.session_state.api_key_valid:
    llm = ChatOpenAI(
        api_key=api_key,
        model_name="gpt-4-turbo-preview",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )


def format_docs(docs):

    return "\n\n".join(document.page_content for document in docs)


st.title("QuizGPT")

if st.session_state.api_key_valid:
    formatting_prompt = PromptTemplate.from_template(
        "{context}의 내용을 바탕으로 퀴즈를 만들어 주세요. 난이도 기준은 매우 쉬움 입니다. 모두 한국어로 만들어 주세요."
    )


if st.session_state.api_key_valid:

    formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {
        "context": format_docs,
    } | formatting_chain
    result = chain.invoke(_docs)

    result_to_json = json.loads(result.additional_kwargs["function_call"]["arguments"])

    return result_to_json


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=1, lang="ko")
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    topic = st.text_input("Search Wikipedia...")
    difficulty_options = ["쉬움", "보통", "어려움"]
    selected_difficulty = st.sidebar.selectbox(
        "퀴즈의 난이도를 선택하세요.", difficulty_options
    )
    if topic:
        docs = wiki_search(topic)

all_correct = True

if not topic or not selected_difficulty:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic)

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
                all_correct = False

        button = st.form_submit_button()

        if button:
            if all_correct:
                st.balloons()


st.sidebar.markdown(
    f'<a href="https://github.com/LontoJ/gpt_study_assignment" target="blank">깃 허브 링크</a>',
    unsafe_allow_html=True,
)
