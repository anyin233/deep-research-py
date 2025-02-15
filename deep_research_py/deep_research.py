from typing import List, Dict, TypedDict, Optional
import asyncio
import os
import openai
from firecrawl import FirecrawlApp
from .ai.providers import trim_prompt
from .prompt import system_prompt
import json
from .common import count_token, log_event, log_error
from .utils import process_response_content
from pydantic import BaseModel


class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class ResearchResult(TypedDict):
    learnings: List[str]
    visited_urls: List[str]


class SerpQuery(BaseModel):
    query: str
    research_goal: str

class SerpQueryResponse(BaseModel):
    queries: List[SerpQuery]


class Firecrawl:
    """Simple wrapper for Firecrawl SDK."""

    def __init__(self, api_key: str = "", api_url: Optional[str] = None):
        self.app = FirecrawlApp(api_key=api_key, api_url=api_url)

    async def search(
        self, query: str, timeout: int = 15000, limit: int = 5
    ) -> SearchResponse:
        """Search using Firecrawl SDK in a thread pool to keep it async."""
        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.app.search(
                    query=query,
                ),
            )

            # Handle the response format from the SDK
            if isinstance(response, dict) and "data" in response:
                # Response is already in the right format
                return response
            elif isinstance(response, dict) and "success" in response:
                # Response is in the documented format
                return {"data": response.get("data", [])}
            elif isinstance(response, list):
                # Response is a list of results
                formatted_data = []
                for item in response:
                    if isinstance(item, dict):
                        formatted_data.append(item)
                    else:
                        # Handle non-dict items (like objects)
                        formatted_data.append(
                            {
                                "url": getattr(item, "url", ""),
                                "markdown": getattr(item, "markdown", "")
                                or getattr(item, "content", ""),
                                "title": getattr(item, "title", "")
                                or getattr(item, "metadata", {}).get("title", ""),
                            }
                        )
                return {"data": formatted_data}
            else:
                print(f"Unexpected response format from Firecrawl: {type(response)}")
                return {"data": []}

        except Exception as e:
            print(f"Error searching with Firecrawl: {e}")
            print(
                f"Response type: {type(response) if 'response' in locals() else 'N/A'}"
            )
            return {"data": []}


# Initialize Firecrawl
firecrawl = Firecrawl(
    api_key=os.environ.get("FIRECRAWL_API_KEY", ""),
    api_url=os.environ.get("FIRECRAWL_BASE_URL"),
)


async def generate_serp_queries(
    query: str,
    client: openai.OpenAI,
    model: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """Generate SERP queries based on user input and previous learnings."""

    prompt = (
        f"Given the following prompt from the user, generate a list of SERP queries to research the topic. "
        f"Return should contain {num_queries} queries (or less if the original prompt is clear). "
        f"Make sure each query is unique and not similar to each other: <prompt>{query}</prompt>"
        f"<schema>{SerpQueryResponse.model_json_schema()}</schema>."
    )

    if learnings:
        prompt += f"\n\nHere are some learnings from previous research, use them to generate more specific queries: {' '.join(learnings)}"

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format=SerpQueryResponse.model_json_schema(),
        ),
    )

    try:
        result = process_response_content(response.choices[0].message.content)
        count_token(response, f"generate_serp_queries: {query[:50]}")
        queries = result.get("queries", [])
        log_event(
            f"generate serp queries: {query}, got {len(queries)}, use queries: {[SerpQuery(**q) for q in queries][:num_queries]}",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return [SerpQuery(**q) for q in queries][:num_queries]
    except json.JSONDecodeError as e:
        log_error(f"Error parsing JSON response: {e}")
        log_error(f"Raw response: {response.choices[0].message.content}")
        return []


class LearningResponse(BaseModel):
    learnings: List[str]
    followUpQuestions: List[str]
    
async def process_serp_result(
    query: str,
    search_result: SearchResponse,
    client: openai.OpenAI,
    model: str,
    num_learnings: int = 3,
    num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """Process search results to extract learnings and follow-up questions."""

    contents = [
        trim_prompt(item.get("markdown", ""), 25_000)
        for item in search_result["data"]
        if item.get("markdown")
    ]

    # Create the contents string separately
    contents_str = "".join(f"<content>\n{content}\n</content>" for content in contents)

    prompt = (
        f"Given the following contents from a SERP search for the query <query>{query}</query>, "
        f"generate a list of learnings from the contents."
        f"Include up to {num_learnings} learnings and "
        f"{num_follow_up_questions} follow-up questions. The learnings should be unique, "
        "concise, and information-dense, including entities, metrics, numbers, and dates.\n\n"
        f"<contents>{contents_str}</contents>"
        f"<schema>{LearningResponse.model_json_schema()}</schema>"
    )

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ],
            response_format=LearningResponse.model_json_schema(),
        ),
    )

    try:
        print(response.choices[0].message.content)
        result = process_response_content(response.choices[0].message.content)
        count_token(response, f"process_serp_result: {query[:50]}")
        log_event(
            f"process serp result: {query}, got {len(result.get('learnings', []))} learnings: {result.get('learnings', [])[:num_learnings]} \n follow up questions: {result.get('followUpQuestions', [])[:num_follow_up_questions]}",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return {
            "learnings": result.get("learnings", [])[:num_learnings],
            "followUpQuestions": result.get("followUpQuestions", [])[
                :num_follow_up_questions
            ],
        }
    except json.JSONDecodeError as e:
        log_error(f"Error parsing JSON response: {e}")
        log_error(f"Raw response: {response.choices[0].message.content}")
        return {"learnings": [], "followUpQuestions": []}


class FinalReportResponse(BaseModel):
    reportMarkdown: str
    
async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str],
    client: openai.OpenAI,
    model: str,
) -> str:
    """Generate final report based on all research learnings."""

    learnings_string = trim_prompt(
        "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
        150_000,
    )

    user_prompt = (
        f"Given the following prompt from the user, write a final report on the topic using "
        f"the learnings from research."
        f"Result should contain a detailed markdown report (aim for 3+ pages). Include ALL the learnings "
        f"from research:\n\n<prompt>{prompt}</prompt>\n\n"
        f"Here are all the learnings from research:\n\n<learnings>\n{learnings_string}\n</learnings>"
        f"<schema>{FinalReportResponse.model_json_schema()}</schema>"
    )

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            response_format=FinalReportResponse.model_json_schema(),
        ),
    )

    try:
        result = process_response_content(response.choices[0].message.content)
        count_token(response, f"write_final_report: {prompt[:50]}")
        report = result.get("reportMarkdown", "")
        log_event(
            f"write final report: {prompt}, got report with {len(report)} characters",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

        # Append sources
        urls_section = "\n\n## Sources\n\n" + "\n".join(
            [f"- {url}" for url in visited_urls]
        )
        return report + urls_section
    except json.JSONDecodeError as e:
        log_error(f"Error parsing JSON response: {e}")
        log_error(f"Raw response: {response.choices[0].message.content}")
        return "Error generating report"


async def deep_research(
    query: str,
    breadth: int,
    depth: int,
    concurrency: int,
    client: openai.OpenAI,
    model: str,
    learnings: List[str] = None,
    visited_urls: List[str] = None,
) -> ResearchResult:
    """
    Main research function that recursively explores a topic.

    Args:
        query: Research query/topic
        breadth: Number of parallel searches to perform
        depth: How many levels deep to research
        learnings: Previous learnings to build upon
        visited_urls: Previously visited URLs
    """
    learnings = learnings or []
    visited_urls = visited_urls or []

    # Generate search queries
    serp_queries = await generate_serp_queries(
        query=query,
        client=client,
        model=model,
        num_queries=breadth,
        learnings=learnings,
    )

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)

    async def process_query(serp_query: SerpQuery) -> ResearchResult:
        async with semaphore:
            try:
                # Search for content
                result = await firecrawl.search(
                    serp_query.query, timeout=15000, limit=5
                )

                # Collect new URLs
                new_urls = [
                    item.get("url") for item in result["data"] if item.get("url")
                ]

                # Calculate new breadth and depth for next iteration
                new_breadth = max(1, breadth // 2)
                new_depth = depth - 1

                # Process the search results
                new_learnings = await process_serp_result(
                    query=serp_query.query,
                    search_result=result,
                    num_follow_up_questions=new_breadth,
                    client=client,
                    model=model,
                )

                all_learnings = learnings + new_learnings["learnings"]
                all_urls = visited_urls + new_urls

                # If we have more depth to go, continue research
                if new_depth > 0:
                    print(
                        f"Researching deeper, breadth: {new_breadth}, depth: {new_depth}"
                    )

                    next_query = f"""
                    Previous research goal: {serp_query.research_goal}
                    Follow-up research directions: {" ".join(new_learnings["followUpQuestions"])}
                    """.strip()

                    return await deep_research(
                        query=next_query,
                        breadth=new_breadth,
                        depth=new_depth,
                        concurrency=concurrency,
                        learnings=all_learnings,
                        visited_urls=all_urls,
                        client=client,
                        model=model,
                    )

                return {"learnings": all_learnings, "visited_urls": all_urls}

            except Exception as e:
                if "Timeout" in str(e):
                    print(f"Timeout error running query: {serp_query.query}: {e}")
                else:
                    print(f"Error running query: {serp_query.query}: {e}")
                return {"learnings": [], "visited_urls": []}

    # Process all queries concurrently
    results = await asyncio.gather(*[process_query(query) for query in serp_queries])

    # Combine all results
    all_learnings = list(
        set(learning.__repr__() for result in results for learning in result["learnings"])
    )

    all_urls = list(set(url for result in results for url in result["visited_urls"]))

    return {"learnings": all_learnings, "visited_urls": all_urls}
