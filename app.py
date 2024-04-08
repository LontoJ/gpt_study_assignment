import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

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
        model_name="gpt-3.5-turbo",
        temperature=0.1,
    )


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


if st.session_state.api_key_valid:
    choose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Use ONLY the following pre-existing answers to answer the user's question.

                Use the answers that have the highest score (more helpful) and favor the most recent ones.

                Cite sources and return the sources of the answers as they are, do not change them.

                Answers: {answers}
                """,
            ),
            ("human", "{question}"),
        ]
    )


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url, filter_value):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[filter_value],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(
            api_key=api_key,
        ),
    )
    return vector_store.as_retriever()


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    choice = st.selectbox(
        "Select an option:",
        options=[None, "AI Gateway", "Cloudflare Vectorize", "Workers AI"],
        index=0,  # 첫 번째 옵션(None)이 기본 선택
        format_func=lambda x: (
            "Please select an option" if x is None else x
        ),  # None 대신 표시할 텍스트
    )

if choice:
    # 선택에 따라 filter_urls의 값 설정
    filter_value = ""
    if choice == "AI Gateway":
        filter_value = r"^(.*\/ai-gateway\/).*"
    elif choice == "Cloudflare Vectorize":
        filter_value = r"^(.*\/vectorize\/).*"
    elif choice == "Workers AI":
        filter_value = r"^(.*\/workers-ai\/).*"
    retriever = load_website(
        "https://developers.cloudflare.com/sitemap.xml", filter_value
    )
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))
