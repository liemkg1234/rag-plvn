import os
from openai import OpenAI

########################################################################################
# Add Contextual Retrieval
__DESCRIPTION = """
Phương pháp: Thêm một ngữ cảnh ngắn gọn để định vị phần này trong toàn bộ tài liệu nhằm mục đích cải thiện khả năng tìm kiếm phần ở chunk:
- Summary contenxt
- Header if chunk dont have
# https://www.anthropic.com/news/contextual-retrieval
Độ dài: 50-100 tokens
"""

_PROMPT = """
<filename>
{FILE_NAME}
</filename>
<document> 
{WHOLE_DOCUMENT}
</document> 
Here is the chunk we want to situate within the whole document:
<chunk> 
{CHUNK_CONTENT}
</chunk> 

Example response:
<example_response>
SUMMARY CONTEXT: Summary context of chunk in the document in {LANGUAGE}

HEADER: 
- <policy_full_name>/<file_name>/<header_of_chunk_1>/<subheader_of_chunk_1>/.../<title_of_chunk_1><subtitle_of_chunk_1>
- <policy_full_name>/<file_name>/<header_of_chunk_2>/<subheader_of_chunk_2>/.../<title_of_chunk_2><subtitle_of_chunk_2>
...

KEYWORDS: <keyword1>, <keyword2>, <keyword3>, ...
</example_response>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Add headers/sub-herders of chunk in document.
Add keywords of the chunk.
Answer only with the succinct context and nothing else. 
""".strip()


def invoke(file_name: str, whole_document: str, chunk_content: str, language: str) -> str:
    client = OpenAI(
        base_url=os.getenv("LLM_GATEWAY_URL"),
        api_key=os.getenv("LLM_LAB_API_KEY"),
    )

    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[
            {
                "role": "system",
                "content": _PROMPT.format(
                    FILE_NAME=file_name,
                    WHOLE_DOCUMENT=whole_document,
                    CHUNK_CONTENT=chunk_content,
                    LANGUAGE=language,
                ),
            },
            {
                "role": "user",
                "content": "Please give a short succinct context, skip the greeting or introduction, just only the content.",
            },
        ],
    )

    answer = completion.choices[0].message.content

    final_chunk = f"{answer}\n\nCONTENT:\n\n{chunk_content}"
    return final_chunk.strip()
