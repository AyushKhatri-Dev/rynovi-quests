"""Day 29 · Chat template by hand

A chat model never sees "messages". It sees one long string with special tokens marking who speaks.
Get the format exactly right and the model behaves; get it wrong and a good model acts dumb.

A message is a dict: {"role": "system" | "user" | "assistant", "content": "..."}
Run `python quest.py 29` after each TODO.
"""

Message = dict[str, str]


def chatml(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """TODO 29.1 · The ChatML format (Qwen and many others). Each message becomes:
        "<|im_start|>{role}\\n{content}<|im_end|>\\n"
    joined together with nothing in between.

    TODO 29.2 · If add_generation_prompt is True, end with "<|im_start|>assistant\\n"
    so the model knows it's its turn to speak.
    """
    # raise NotImplementedError
    st = ""
    for msg in messages:
        st += f"<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>\n"
        if add_generation_prompt:
            st += "<|im_start|>assistant\n"
        
    return st 


def llama3(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """TODO 29.3 · The Llama 3 format. Start with "<|begin_of_text|>", then each message is:
        "<|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>"
    With add_generation_prompt, end with "<|start_header_id|>assistant<|end_header_id|>\\n\\n".
    """
    st = "<|begin_of_text|>"
    for msg in messages:
        st += f"<|start_header_id|>{msg["role"]}<|end_header_id|>\n\n{msg["content"]}<|eot_id|>"
    if add_generation_prompt:
        st += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return st


def check_conversation(messages: list[Message]) -> None:
    """TODO 29.4 · Raise ValueError if the conversation is malformed:
      - a role other than system, user or assistant
      - a system message anywhere except first
      - after the optional system message, turns must go user, assistant, user, assistant... starting with user
    Return None if it's fine.
    """
    rl = ["system", "user", "assistant"]
    for i, msg in enumerate(messages):
        if msg["role"] not in rl:
                raise ValueError(f"unknown role: {msg['role']}")
        if msg["role"] == "system" and i != 0:
            raise ValueError("system message must be first")
        if messages and messages[0]["role"] == "system":
            turns = messages[1:]
        else:
            turns = messages 
        for i, msg in enumerate(turns):
            if i % 2 == 0:
                expected = "user"
            else:
                expected = "assistant"

            if msg["role"] != expected:
                raise ValueError(f"turn {i} should be {expected}, got {msg['role']}")