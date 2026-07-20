from __future__ import annotations as _annotations

import httpx
import pytest

from pydantic_ai import ModelAPIError, ModelRequest, TextPart
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


async def test_stream_clean_eof_without_finish_reason(allow_model_requests: None):
    sse = b"""data: {"id":"123","choices":[{"index":0,"delta":{"content":"partial response","role":"assistant"},"finish_reason":null}],"created":1704067200,"model":"gpt-4o-123","object":"chat.completion.chunk"}\n\n"""

    def handle_request(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=sse)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as http_client:
        client = AsyncOpenAI(api_key='test', http_client=http_client)
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))

        async with model.request_stream(
            [ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters()
        ) as stream:
            event_count = 0
            with pytest.raises(ModelAPIError, match='Streamed response ended without a finish reason') as exc_info:
                async for _ in stream:
                    event_count += 1

            response = stream.get()

    assert event_count > 0
    assert exc_info.value.model_name == 'gpt-4o-123'
    assert response.parts == [TextPart(content='partial response')]
    assert response.state == 'incomplete'
    assert response.finish_reason is None
