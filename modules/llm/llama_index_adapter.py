import os
import sys
from typing import Any, List, Optional, Dict, Mapping, Sequence, Generator, Union
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse, CompletionResponse
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.memory import ChatMemoryBuffer

from .local_llm import LocalLLM
from pydantic import PrivateAttr

class LlamaIndexAdapter(CustomLLM):
    """
    Adapt LocalLLM to LlamaIndex framework with built-in conversation history management
    """

    history_window: int = 3
    _local_llm: "LocalLLM" = PrivateAttr()
    _memory: "ChatMemoryBuffer" = PrivateAttr()

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        token: Optional[str] = None,
        history_window: int = 3,  # Keep last 3 rounds of conversation
    ):

        super().__init__()

        self.history_window = history_window

        self._local_llm = LocalLLM(
            model_name_or_path=model_name_or_path,
            device=device,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Initialize chat memory with token limit or message window
        self._memory = ChatMemoryBuffer.from_defaults(
            token_limit=2048,  # Adjust based on your model's context window
        )

    @property
    def local_llm(self) -> "LocalLLM":
        return self._local_llm

    @property
    def memory(self) -> "ChatMemoryBuffer":
        return self._memory


    def _convert_message_to_dict(self, message: ChatMessage) -> Dict[str, str]:
        """Convert LlamaIndex message to local LLM message format"""
        return {
            "role": message.role.value,
            "content": message.content
        }

    def _convert_messages_to_dicts(self, messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
        """Convert LlamaIndex message list to local LLM message dictionary list"""
        return [self._convert_message_to_dict(msg) for msg in messages]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": getattr(self.local_llm, "model_name_or_path", "local_llm"),
            "context_window": 4096,
        }

    def complete(self, prompt: str, **kwargs) -> str:
        """Complete prompt (single prompt text)"""
        # Convert to message format
        messages = [{"role": "user", "content": prompt}]

        # Use local LLM to generate response
        response = self.local_llm.generate_response(messages, **kwargs)

        return response

    def generate_response(self, messages, system_prompt=None, temperature=None, max_tokens=None):
        # 1) Get the last user message
        last_user = None
        for m in reversed(messages or []):
            if (m.get("role") or "").lower() == "user":
                last_user = m.get("content", "")
                break
        if last_user:
            self.memory.put(ChatMessage(role=MessageRole.USER, content=last_user))

        # 2) Retrieve history from memory
        mem_msgs = self.memory.get()  # List[ChatMessage]
        mem_dicts = [{"role": m.role.value, "content": m.content} for m in mem_msgs]

        # 3) Retain only the last N rounds (N = history_window)
        #    One round = user + assistant; system messages are not counted
        core = [m for m in mem_dicts if m["role"] != "system"]
        window = []
        rounds = 0
        i = len(core) - 1
        while i >= 0 and rounds < self.history_window:
            cur = []
            while i >= 0 and core[i]["role"] == "assistant":
                cur.append(core[i]);
                i -= 1
            if i >= 0 and core[i]["role"] == "user":
                cur.append(core[i]);
                i -= 1
                rounds += 1
                window.extend(reversed(cur))
            else:
                break
        window = list(reversed(window))

        # 4) Pass to local model
        text = self.local_llm.generate_response(
            window, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
        )

        # 5) Write back the assistant's response
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=text))
        return text

    def stream_complete(self, prompt: str, **kwargs) -> Generator[CompletionResponse, None, None]:
        """Stream completion prompt (currently doesn't support true streaming, returns complete response)"""
        # Get complete response
        full_response = self.complete(prompt, **kwargs)

        # Return complete response as one-time stream
        completion_response = CompletionResponse(text=full_response)
        yield completion_response

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        """
        Chat (multi-turn conversation) with LlamaIndex's built-in memory management
        """
        # Convert messages format
        converted_messages = self._convert_messages_to_dicts(messages)

        # Extract system prompt (if any)
        system_prompt = None
        filtered_messages = []

        for msg in converted_messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)

        # Get chat history from memory
        chat_history = self.memory.get()
        history_messages = []

        if chat_history:
            for hist_msg in chat_history:
                if isinstance(hist_msg, ChatMessage):
                    history_messages.append(self._convert_message_to_dict(hist_msg))

        # Combine history with current messages
        all_messages = history_messages + filtered_messages

        # Keep only the last N rounds (history_window)
        if len(all_messages) > self.history_window * 2:
            all_messages = all_messages[-(self.history_window * 2):]

        # Use local LLM to generate response
        response_text = self.local_llm.generate_response(
            all_messages,
            system_prompt=system_prompt,
            **kwargs
        )

        # Create ChatMessage for response
        response_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response_text
        )

        # Update memory with the conversation
        for msg in messages:
            self.memory.put(msg)
        self.memory.put(response_message)

        # Create ChatResponse object
        return ChatResponse(
            message=response_message
        )

    def reset_conversation(self):
        """Reset conversation history"""
        self.memory.reset()
        self.conversation_history.clear()