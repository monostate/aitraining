"""
Format-specific Message Renderers
==================================

Implementations for various chat formats.
"""

from typing import List

from .message_renderer import Conversation, Message, MessageRenderer


class VicunaRenderer(MessageRenderer):
    """Renderer for Vicuna format."""

    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation in Vicuna format."""
        rendered = []

        # Handle system message
        system_msg = ""
        for msg in conversation.messages:
            if msg.role == "system":
                system_msg = msg.content + "\n\n"
                break

        if system_msg:
            rendered.append(system_msg)

        # Process user/assistant pairs
        messages = [m for m in conversation.messages if m.role != "system"]

        for i, msg in enumerate(messages):
            if msg.role == "user":
                rendered.append(f"USER: {msg.content}")
            elif msg.role == "assistant":
                if msg.content:
                    rendered.append(f"ASSISTANT: {msg.content}")
                else:
                    # For generation
                    rendered.append("ASSISTANT:")

        return " ".join(rendered)

    def _render_single_message(self, message: Message) -> str:
        """Render single Vicuna message."""
        if message.role == "user":
            return f"USER: {message.content}"
        elif message.role == "assistant":
            return f"ASSISTANT: {message.content}"
        else:
            return message.content

    def get_stop_sequences(self) -> List[str]:
        """Get Vicuna stop sequences."""
        return ["USER:", "ASSISTANT:", "</s>"]

    def parse_response(self, response: str) -> str:
        """Parse Vicuna response."""
        response = response.replace("ASSISTANT:", "")
        if "USER:" in response:
            response = response.split("USER:")[0]
        return response.strip()


class ZephyrRenderer(MessageRenderer):
    """Renderer for Zephyr format."""

    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation in Zephyr format."""
        rendered = []

        for msg in conversation.messages:
            if msg.role == "system":
                rendered.append(f"<|system|>\n{msg.content}</s>")
            elif msg.role == "user":
                rendered.append(f"<|user|>\n{msg.content}</s>")
            elif msg.role == "assistant":
                if msg.content:
                    rendered.append(f"<|assistant|>\n{msg.content}</s>")
                else:
                    # For generation
                    rendered.append("<|assistant|>\n")

        return "\n".join(rendered)

    def _render_single_message(self, message: Message) -> str:
        """Render single Zephyr message."""
        if message.content:
            return f"<|{message.role}|>\n{message.content}</s>"
        else:
            return f"<|{message.role}|>\n"

    def get_stop_sequences(self) -> List[str]:
        """Get Zephyr stop sequences."""
        return ["</s>", "<|user|>", "<|assistant|>", "<|system|>"]

    def parse_response(self, response: str) -> str:
        """Parse Zephyr response."""
        # Remove role markers
        for marker in ["<|assistant|>", "<|user|>", "<|system|>"]:
            response = response.replace(marker, "")
        response = response.replace("</s>", "")

        return response.strip()


class LlamaRenderer(MessageRenderer):
    """Renderer for Llama format (Llama-2/3 style)."""

    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation in Llama format."""
        rendered = []

        # Build system prompt
        system_prompt = "You are a helpful assistant."
        for msg in conversation.messages:
            if msg.role == "system":
                system_prompt = msg.content
                break

        # Process conversations
        messages = [m for m in conversation.messages if m.role != "system"]

        for i, msg in enumerate(messages):
            if msg.role == "user":
                if i == 0:
                    # Include system prompt with first user message
                    rendered.append(f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{msg.content} [/INST]")
                else:
                    rendered.append(f"<s>[INST] {msg.content} [/INST]")
            elif msg.role == "assistant":
                if msg.content:
                    rendered.append(f" {msg.content} </s>")
                else:
                    # For generation
                    rendered.append(" ")

        return "".join(rendered)

    def _render_single_message(self, message: Message) -> str:
        """Render single Llama message."""
        if message.role == "user":
            return f"[INST] {message.content} [/INST]"
        elif message.role == "assistant":
            return f" {message.content} </s>"
        else:
            return f"<<SYS>>\n{message.content}\n<</SYS>>"

    def get_stop_sequences(self) -> List[str]:
        """Get Llama stop sequences."""
        return ["</s>", "[INST]", "[/INST]"]

    def parse_response(self, response: str) -> str:
        """Parse Llama response."""
        response = response.replace("</s>", "")
        if "[INST]" in response:
            response = response.split("[INST]")[0]
        return response.strip()


class MistralRenderer(MessageRenderer):
    """Renderer for Mistral format."""

    def render_conversation(self, conversation: Conversation) -> str:
        """Render conversation in Mistral format."""
        rendered = []

        for msg in conversation.messages:
            if msg.role == "user":
                rendered.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == "assistant":
                if msg.content:
                    rendered.append(msg.content)
                else:
                    # For generation, just end with the instruction tag
                    pass
            elif msg.role == "system":
                # Mistral typically includes system in first user message
                pass

        # Handle system message by prepending to first user message
        system_content = None
        for msg in conversation.messages:
            if msg.role == "system":
                system_content = msg.content
                break

        if system_content and rendered:
            # Modify first instruction to include system
            first_inst = rendered[0]
            if first_inst.startswith("[INST]"):
                content = first_inst[6:-7]  # Remove [INST] and [/INST]
                rendered[0] = f"[INST] {system_content}\n\n{content} [/INST]"

        return " ".join(rendered)

    def _render_single_message(self, message: Message) -> str:
        """Render single Mistral message."""
        if message.role == "user":
            return f"[INST] {message.content} [/INST]"
        elif message.role == "assistant":
            return message.content
        else:
            return message.content

    def get_stop_sequences(self) -> List[str]:
        """Get Mistral stop sequences."""
        return ["[INST]", "[/INST]", "</s>"]

    def parse_response(self, response: str) -> str:
        """Parse Mistral response."""
        if "[INST]" in response:
            response = response.split("[INST]")[0]
        response = response.replace("</s>", "")
        return response.strip()


# Registry of available formats
AVAILABLE_FORMATS = {
    "chatml": "ChatMLRenderer",
    "alpaca": "AlpacaRenderer",
    "vicuna": "VicunaRenderer",
    "zephyr": "ZephyrRenderer",
    "llama": "LlamaRenderer",
    "llama2": "LlamaRenderer",
    "llama3": "LlamaRenderer",
    "mistral": "MistralRenderer",
}


__all__ = [
    "VicunaRenderer",
    "ZephyrRenderer",
    "LlamaRenderer",
    "MistralRenderer",
    "AVAILABLE_FORMATS",
]
