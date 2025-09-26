#!/bin/bash
cd /home/kavia/workspace/code-generation/openai-chat-assistant-21646-21655/llm_chat_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

