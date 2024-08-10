import logging
import os

from langchain.output_parsers import YamlOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain_anthropic import ChatAnthropic

# Initialize the LLM
chat_anthropic = ChatAnthropic(
    temperature=0,
    model_name="claude-3-5-sonnet-20240620",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=30,
    max_retries=0,
)

llm = chat_anthropic


def invoke_llm(model, prompt, *args, **kwargs):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            *[("human", arg) for arg in args],
        ]
    )

    parser = YamlOutputParser(pydantic_object=model)

    # Combine the prompt with the structured LLM runnable
    chain = chat_prompt | llm | parser

    # Invoke the runnable to get structured output
    result = chain.invoke(kwargs)
    logging.info(f"{result=}")
    return result
