---
title: Effective AI Chatbot
emoji: 🌟
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.26.0
python_version: "3.13"
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - inference-api
---

# 🌟 Effective AI Chatbot

A fancier version of the standard Hugging Face chatbot template, built with
[Gradio](https://www.gradio.app/). It talks to a hosted model through the
Hugging Face Inference API and falls back to a small model running locally when
the API call fails.

> Note: this is a practice project for DS553, not an official app.

## How it works

- **Remote model** (`openai/gpt-oss-20b`): used by default. Requests are made
  with your Hugging Face token via `InferenceClient.chat_completion`, streamed
  back token by token.
- **Local fallback** (`Qwen/Qwen3-0.6B`): loaded through a `transformers`
  pipeline and used automatically if the remote call raises.

## Authentication

The app uses Hugging Face OAuth. On the deployed Space, click the **Sign in with
Hugging Face** button in the sidebar to grant an `inference-api` token.

Running locally, Gradio mocks OAuth, so provide a real token instead:

```bash
export HF_TOKEN=hf_...      # or:
hf auth login
```

`resolve_hf_token()` in `app.py` prefers the OAuth token and falls back to
`HF_TOKEN` / the CLI login cache.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the printed local URL.

## Configuration

Adjust the system prompt, max new tokens, temperature, and top-p from the
**Additional inputs** panel in the chat UI.

## Tests

```bash
pytest
```
