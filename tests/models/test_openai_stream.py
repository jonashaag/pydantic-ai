from __future__ import annotations as _annotations

import os

import pytest

from pydantic_ai import ModelMessage, ModelRequest
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture(scope='session')
def evroc_api_key() -> str:
    return os.getenv('EVROC_API_KEY', 'mock-api-key')


async def test_clean_eof_without_finish_reason(allow_model_requests: None, evroc_api_key: str):
    """Reproduce clean EOF using one content chunk from a real evroc recording."""
    model = OpenAIChatModel(
        'moonshotai/Kimi-K2.6',
        provider=OpenAIProvider(base_url='https://models.think.evroc.com/v1', api_key=evroc_api_key),
    )
    settings = OpenAIChatModelSettings(
        extra_headers={'X-Think-Timeout': '1'},
        extra_body={'chat_template_kwargs': {'thinking': False}},
    )
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Count to 100000, one number per line.')]

    async with model.request_stream(messages, settings, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass

    response = stream.get()
    assert response.state == 'complete'
    assert response.finish_reason is None
    assert response.text == ' I'
