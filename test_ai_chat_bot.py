import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain

model = ChatTongyi(
    model_name="qwen-max",
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY'),
    model_kwargs={"temperature": 0.001},
    streaming=True
)



memory_key = "history"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful chatbot"),
        MessagesPlaceholder(variable_name=memory_key),
        ("user", "{input}")
    ]
)

memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

conversation_chain = ConversationChain(
    llm=model,
    prompt=prompt,
    memory=memory
)


while True:
    user_input = input("User: ")
    if user_input == "quit":
        break
    print(f"{memory.load_memory_variables({})}")
    print(f"AI: ", end="")
    for chunk in conversation_chain.stream({"input": user_input}):
        print(chunk["response"], end="", flush=True)
    print()


