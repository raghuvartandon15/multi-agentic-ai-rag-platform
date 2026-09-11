from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def detect_input_type(user_input):
    if isinstance(user_input, str):
        return "text"

    if isinstance(user_input, Path):
        suffix = user_input.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix in [".jpg", ".jpeg", ".png", ".webp"]:
            return "image"
    return "unknown"


def handle_text(text: str):
    cleaned_text = " ".join(text.split())
    return cleaned_text


def handle_pdf(file_path: Path):
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    extracted_text = "\n".join(pages)
    return extracted_text


def handle_image(file_path):

    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage
    import base64
    from input_handler.config import MODEL, MODEL_PROVIDER

    llm = init_chat_model(model_provider=MODEL_PROVIDER,model=MODEL, max_tokens=900, reasoning_effort='none')

    with open(file_path, "rb") as image_file:
        image_data = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    image_suffix = file_path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    mime_type = mime_types.get(image_suffix,"image/png")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """
            Analyze this technical image and produce a concise description
            for downstream AI agents.

            Include:
            - Main components
            - Important labels/text
            - Connections and data flow
            - Architecture structure
            - Obvious technical issues, if any

            Focus on factual visual information.
            Do not explain your reasoning.
            Do not answer any user question.
            Return only the final description.
        """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            }
        ]
    )

    response = llm.invoke([message])

    return response.content


def process_input(user_input):
    input_type = detect_input_type(user_input)

    if input_type == "text":
        content = handle_text(user_input)

    elif input_type == "pdf":
        content = handle_pdf(user_input)

    elif input_type == "image":
        content = handle_image(user_input)

    else:
        raise ValueError(
            "Unsupported input type."
        )

    return {
        "input_type": input_type,
        "raw_input": user_input,
        "extracted_content": content,
    }