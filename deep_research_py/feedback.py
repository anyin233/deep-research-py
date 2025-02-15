from typing import List
import asyncio
import openai
import json
from .prompt import system_prompt
from .common import count_token, log_event
from .utils import process_response_content
from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    questions: List[str]

async def generate_feedback(query: str, client: openai.OpenAI, model: str) -> List[str]:
    """Generates follow-up questions to clarify research direction."""

    # Run OpenAI call in thread pool since it's synchronous
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {
                    "role": "user",
                    "content": "Given the following query from the user, ask some follow up questions to clarify the research direction. Return a maximum of {max_questions} questions in JSON, but feel free to return none if the original query is clear: <query>{query}</query>. You should follow this JSON schema <schema>{schema}</schema>".format(max_questions=max_questions, query=query, schema=FeedbackResponse.model_json_schema()),
                },
            ],
            response_format=FeedbackResponse.model_json_schema(),
        ),
    )


    # Parse the JSON response
    try:
        result = process_response_content(response.choices[0].message.content)
        count_token(response, f"feedback: {query[:50]}")
        if not isinstance(result, list):
            result = result.get('questions', [])
        log_event(
            "generate feedback: {}, got {} questions: {}".format(query, len(result), '\n'.join(result)),
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return []
