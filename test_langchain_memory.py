from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# 单个字符串返回
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("您好")
memory.chat_memory.add_ai_message("您好啊")
print(memory.load_memory_variables({}))

prompt = PromptTemplate.from_template("聊天历史：\n{history}\n")
print(prompt.format(**memory.load_memory_variables({})))


print("#########################################")
# 消息列表返回
memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("您好")
memory.chat_memory.add_ai_message("您好啊")
print(memory.load_memory_variables({}))

prompt = PromptTemplate.from_template("聊天历史：\n{history}\n")
print(prompt.format(**memory.load_memory_variables({})))