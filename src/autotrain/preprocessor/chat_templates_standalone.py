"""
Standalone chat templates extracted from Unsloth.
This file works without GPU dependencies.

Source: Extracted from unsloth/chat_templates.py
Total templates: 32
"""

# Chat templates dictionary
CHAT_TEMPLATES = {}

# alpaca template
CHAT_TEMPLATES["alpaca"] = (
    """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n\n' }}{% set loop_messages = messages[1:] %}{% else %}{{ '{system_message}' + '\n\n' }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '### Instruction:\n' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Response:\n' + message['content'] + eos_token + '\n\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Response:\n' }}{% endif %}"""
)

# chatml template
CHAT_TEMPLATES["chatml"] = (
    """{% for message in messages %}{% if message['role'] == 'user' %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% else %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
)

# gemma template
CHAT_TEMPLATES["gemma"] = (
    """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{'<start_of_turn>user\n' + messages[0]['content'] | trim + ' ' + messages[1]['content'] | trim + '<end_of_turn>\n'}}{% set messages = messages[2:] %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{'<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n'}}{% elif message['role'] == 'assistant' %}{{'<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"""
)

# gemma-3 template
CHAT_TEMPLATES[
    "gemma-3"
] = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix =  -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("""

# gemma-3n template
CHAT_TEMPLATES[
    "gemma-3n"
] = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix =  -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("""

# gemma2 template
CHAT_TEMPLATES["gemma2"] = (
    """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{'<start_of_turn>user\n' + messages[0]['content'] | trim + ' ' + messages[1]['content'] | trim + '<end_of_turn>\n'}}{% set messages = messages[2:] %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{'<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n'}}{% elif message['role'] == 'assistant' %}{{'<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"""
)

# gemma3 template
CHAT_TEMPLATES[
    "gemma3"
] = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix =  -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("""

# gemma3n template
CHAT_TEMPLATES[
    "gemma3n"
] = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix =  -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("""

# gpt-oss template
CHAT_TEMPLATES[
    "gpt-oss"
] = """{#-
  In addition to the normal inputs of `messages` and `tools`, this template also accepts the
  following kwargs:
  - """

# gptoss template
CHAT_TEMPLATES[
    "gptoss"
] = """{#-
  In addition to the normal inputs of `messages` and `tools`, this template also accepts the
  following kwargs:
  - """

# llama template
CHAT_TEMPLATES["llama"] = (
    """{% if messages[0]['role'] == 'system' %}{% if messages[1]['role'] == 'user' %}{{ bos_token + '[INST] <<SYS>>\n' + messages[0]['content'] + '\n<</SYS>>\n\n' + messages[1]['content'] + ' [/INST]' }}{% set loop_messages = messages[2:] %}{% else %}{{ bos_token + '[INST] ' + messages[0]['content'] + ' [/INST]' }}{% set loop_messages = messages[1:] %}{% endif %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'].strip() + ' ' + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""
)

# llama-3 template
CHAT_TEMPLATES["llama-3"] = (
    """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% else %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""
)

# llama-3.1 template
CHAT_TEMPLATES[
    "llama-3.1"
] = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = """

# llama-31 template
CHAT_TEMPLATES[
    "llama-31"
] = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = """

# llama3 template
CHAT_TEMPLATES["llama3"] = (
    """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% else %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""
)

# mistral template
CHAT_TEMPLATES["mistral"] = (
    """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% if messages[1]['role'] == 'user' %}{{ '[INST] ' + messages[0]['content'] + ' ' + messages[1]['content'] + ' [/INST]' }}{% set loop_messages = messages[2:] %}{% else %}{{ '[INST] ' + messages[0]['content'] + ' [/INST]' }}{% set loop_messages = messages[1:] %}{% endif %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""
)

# phi-3 template
CHAT_TEMPLATES["phi-3"] = (
    """{% for message in messages %}{% if message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% else %}{{'<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"""
)

# phi-4 template
CHAT_TEMPLATES["phi-4"] = (
    """{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"""
)

# qwen-2.5 template
CHAT_TEMPLATES[
    "qwen-2.5"
] = """{%- if tools %}
    {{- \'<|im_start|>system\\n\' }}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- messages[0][\'content\'] }}
    {%- else %}
        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}
    {%- endif %}
    {{- """

# qwen-25 template
CHAT_TEMPLATES[
    "qwen-25"
] = """{%- if tools %}
    {{- \'<|im_start|>system\\n\' }}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- messages[0][\'content\'] }}
    {%- else %}
        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}
    {%- endif %}
    {{- """

# qwen-3 template
CHAT_TEMPLATES[
    "qwen-3"
] = """
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- """

# qwen2.5 template
CHAT_TEMPLATES[
    "qwen2.5"
] = """{%- if tools %}
    {{- \'<|im_start|>system\\n\' }}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- messages[0][\'content\'] }}
    {%- else %}
        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}
    {%- endif %}
    {{- """

# qwen25 template
CHAT_TEMPLATES[
    "qwen25"
] = """{%- if tools %}
    {{- \'<|im_start|>system\\n\' }}
    {%- if messages[0][\'role\'] == \'system\' %}
        {{- messages[0][\'content\'] }}
    {%- else %}
        {{- \'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\' }}
    {%- endif %}
    {{- """

# qwen3 template
CHAT_TEMPLATES[
    "qwen3"
] = """
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- """

# qwen3-instruct template
CHAT_TEMPLATES[
    "qwen3-instruct"
] = """# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\: <function-name>, \\: <args-json-object>}\\n</tool_call><|im_end|>\\nuserusersystemassistantname' }}
                {{- tool_call.name }}
                {{- 'argumentstooltooltool"""

# qwen3-thinking template
CHAT_TEMPLATES[
    "qwen3-thinking"
] = """# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\\n\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\: <function-name>, \\: <args-json-object>}\\n</tool_call><|im_end|>\\nuserusersystemassistantname' }}
                {{- tool_call.name }}
                {{- 'argumentstooltooltool"""

# starling template
CHAT_TEMPLATES[
    "starling"
] = """{{ bos_token }}
{%- for message in messages %}
    {{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{ 'GPT4 Correct Assistant:' }}
{%- endif %}"""

# unsloth template
CHAT_TEMPLATES["unsloth"] = (
    """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% set loop_messages = messages[1:] %}{% else %}{{ '{system_message}' + '\n' }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '>>> User: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '>>> Assistant: ' }}{% endif %}"""
)

# vicuna template
CHAT_TEMPLATES["vicuna"] = (
    """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + ' ' }}{% set loop_messages = messages[1:] %}{% else %}{{ '{system_message}' + ' ' }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""
)

# vicuna_old template
CHAT_TEMPLATES["vicuna_old"] = (
    """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '\n' }}{% set loop_messages = messages[1:] %}{% else %}{{ '{system_message}' + '\n' }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '### Human: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant: ' + message['content'] + eos_token + '\n' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Assistant:' }}{% endif %}"""
)

# yi-chat template
CHAT_TEMPLATES[
    "yi-chat"
] = """
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}
"""

# zephyr template
CHAT_TEMPLATES["zephyr"] = (
    """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}{% else %}{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"""
)


def get_chat_template(template_name):
    """
    Get a chat template by name.

    Args:
        template_name: Name of the template (e.g., 'llama3', 'chatml', 'alpaca')

    Returns:
        The template string if found, None otherwise
    """
    return CHAT_TEMPLATES.get(template_name)


def list_templates():
    """
    List all available template names.

    Returns:
        Sorted list of template names
    """
    return sorted(CHAT_TEMPLATES.keys())


def get_template_for_model(model_name):
    """
    Get suggested template based on model name.

    Args:
        model_name: Name or path of the model

    Returns:
        Suggested template name or None
    """
    model_lower = model_name.lower() if model_name else ""

    # Direct matches
    if "llama-3" in model_lower or "llama3" in model_lower:
        return "llama3"
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return "llama"
    elif "gemma-2" in model_lower or "gemma2" in model_lower:
        return "gemma2" if "gemma2" in CHAT_TEMPLATES else "gemma"
    elif "gemma" in model_lower:
        return "gemma"
    elif "mistral" in model_lower:
        return "mistral"
    elif "phi-3" in model_lower or "phi3" in model_lower:
        return "phi-3"
    elif "phi-4" in model_lower or "phi4" in model_lower:
        return "phi-4"
    elif "qwen2.5" in model_lower or "qwen-2.5" in model_lower:
        return "qwen2.5"
    elif "qwen" in model_lower:
        return "qwen3"
    elif "vicuna" in model_lower:
        return "vicuna"
    elif "alpaca" in model_lower:
        return "alpaca"
    elif "chatml" in model_lower:
        return "chatml"
    elif "zephyr" in model_lower:
        return "zephyr"
    elif "yi-" in model_lower and "chat" in model_lower:
        return "yi-chat"

    return None
