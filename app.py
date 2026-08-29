import os

import torch
import gradio as gr
#import spaces
from huggingface_hub import InferenceClient, get_token
from transformers import pipeline

LOCAL_MODEL = "Qwen/Qwen3-0.6B"
REMOTE_MODEL = "openai/gpt-oss-20b"

_pipe = None

#install huggingface-cli
#hf auth login
#then pt

def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text-generation",
            model=LOCAL_MODEL,
            dtype="auto",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _pipe

fancy_css = """
.gradio-container {
    width: 96% !important;
    max-width: none !important;
}
#app-title {
    text-align: center;
    margin-bottom: 4px;
}
#app-subtitle {
    text-align: center;
    color: var(--body-text-color-subdued);
    margin-bottom: 24px;
}
#chat-container {
    width: 100%;
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}
#model-note {
    font-size: 0.9em;
    color: var(--body-text-color-subdued);
    margin-top: 8px;
}
@media (max-width: 768px) {
    .gradio-container {
        width: 98% !important;
    }
    #chat-container {
        padding: 8px;
    }
}
"""


def resolve_hf_token(oauth_token):
    """Return a usable Hugging Face token.

    On a deployed Space, `gr.LoginButton` performs a real OAuth handshake and
    `oauth_token.token` is a genuine `hf_...` token. Running locally, Gradio
    mocks OAuth and hands back `mock-oauth-token-for-local-dev`, which the
    InferenceClient rejects ("Cannot select auto-router when using non-Hugging
    Face API key"). In that case fall back to a real token from the environment
    or the CLI login cache (`hf auth login`).
    """
    tok = getattr(oauth_token, "token", None)
    if tok and tok.startswith("hf_"):
        return tok
    return os.environ.get("HF_TOKEN") or get_token()


#@spaces.GPU
def local_generate(
    messages,
    max_tokens,
    temperature,
    top_p,
):
    outputs = get_pipe()(
        messages,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    return outputs[0]["generated_text"][-1]["content"]


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    print(hf_token)

    try:
        print("[MODE] api")

        token = resolve_hf_token(hf_token)
        if not token:
            yield (
                "⚠️ No Hugging Face token available. Log in with the button, "
                "or set HF_TOKEN / run `hf auth login` when developing locally."
            )
            return

        client = InferenceClient(
            token=token,
            model=REMOTE_MODEL,
        )

        response = ""

        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""

            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content

            response += token
            yield response
    except Exception as e:
        print(e)

        response = local_generate(
            messages,
            max_tokens,
            temperature,
            top_p,
        )

        yield response
        return


chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(
            value="You are a friendly Chatbot.",
            label="System message",
        ),
        gr.Slider(
            minimum=1,
            maximum=2048,
            value=512,
            step=1,
            label="Max new tokens",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.7,
            step=0.1,
            label="Temperature",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


with gr.Blocks(css=fancy_css) as demo:
    with gr.Sidebar():
        gr.LoginButton()

    gr.Markdown(
        "# 🌟 Effective AI Chatbot",
        elem_id="app-title",
    )

    gr.Markdown(
        "A fancier version of the standard Huggging Face chatbot template.",
        elem_id="app-subtitle",
    )

    with gr.Column(elem_id="chat-container"):
        chatbot.render()

        gr.Markdown(
            "Use **Additional inputs** to switch between the API model and the locally executed model.",
            elem_id="model-note",
        )


if __name__ == "__main__":
    demo.launch()