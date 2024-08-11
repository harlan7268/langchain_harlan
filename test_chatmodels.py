import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Your name is {name}."),
        ("human", "Hi"),
        ("ai", "Hi."),
        ("human", "{user_input}")
    ]
)

messages = chat_template.format_messages(
    name="Mr.Wang",
    user_input="What's your name?"
)

print(messages)

human_message_prompt_template = HumanMessagePromptTemplate.from_template("{text}")

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You like jokes"
            )
        ),
        human_message_prompt_template,
    ]
)


chat_messages = chat_template.format_messages(text="The weather is great today")
print(chat_messages)


chat_template_final = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a translator, and your task is to translate English into Chinese."
            )
        ),
        HumanMessage(content="Hello."),
        AIMessage(content="你好"),
        HumanMessagePromptTemplate.from_template("{text}")
    ]
)

model = ChatTongyi(
    model_name="qwen-max"
)

chain_first = chat_template_final | model
a = chain_first.invoke({"text": "I like playing football."})
print(a) 

