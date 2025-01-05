import logging
import os

from langchain.output_parsers import YamlOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the LLM
chat_anthropic = ChatAnthropic(
    temperature=0,
    model_name="claude-3-5-sonnet-20241022",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=30,
    max_retries=0,
)

gemini_models = [
    "gemini-exp-1206",
    "gemini-2.0-flash-exp",
    # "gemini-2.0-flash-thinking-exp-1219",
]

chat_gemini_models = [
    ChatGoogleGenerativeAI(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        model=model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        timeout=90,
        max_retries=0,
    )
    for model in gemini_models
]

llm_primary = chat_gemini_models[0].with_fallbacks(chat_gemini_models[1:])
llm_fallback = chat_gemini_models[1]


def invoke_llm(model, prompt, *args, with_fallback=False, **kwargs):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            *[("human", arg) for arg in args],
        ]
    )
    llm = llm_fallback if with_fallback else llm_primary

    parser = YamlOutputParser(pydantic_object=model)

    # Combine the prompt with the structured LLM runnable
    chain = chat_prompt | llm | parser

    # Invoke the runnable to get structured output
    result = chain.invoke(kwargs)
    logging.info(f"{with_fallback=}: {result=}")
    return result
