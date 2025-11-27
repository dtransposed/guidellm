
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Optional, Any

from guidellm.backends.response_handlers import GenerationResponseHandlerFactory
import httpx
import asyncio
import json
from guidellm.backends.backend import Backend
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
)

API_ROUTES = {
    "text_completions": "/service/v5/generate/v1",
    "health": "/health",
}
EXPECTED_REQUEST_FIELDS = ['filepath', 'prefix', 'editable_region_prefix', 'editable_region_suffix', 'suffix', 'history']




@Backend.register("next_edit_suggestion")
class NextEditSuggestionBackend(Backend):
    def __init__(self, target: str, model: Optional[str] = None, api_routes: dict[str, str] = API_ROUTES):
        super().__init__(type_="next_edit_suggestion")
        self.api_routes = api_routes
        self.target = target

    async def process_startup(self):
        self._async_client = httpx.AsyncClient()

    async def process_shutdown(self):
        pass
    
    @property
    def info(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "type": self.type_,
            "api_routes": self.api_routes,
        }

    async def validate(self):
        response = await self._async_client.get(self.target + self.api_routes['health'])
        response.raise_for_status()
        
    async def default_model(self):
        return None
    
    async def resolve(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: Optional[list[tuple[GenerationRequest, GenerationResponse]]] = None,
    ) -> AsyncIterator[tuple[GenerationResponse, RequestInfo]]:
        """
        Process generation request and yield progressive responses.
        
        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :yields: Tuples of (response, updated_request_info) as generation progresses
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        
        if history is not None:
            raise NotImplementedError("Multi-turn requests not yet supported")
        
        request_path = self.api_routes.get(request.request_type)
        if request_path is None:
            raise ValueError(
                f"Unsupported request type '{request.request_type}'. "
                f"Supported types: {list(self.api_routes.keys())}"
            )
        
        request_url = f"{self.target}{request_path}"
        # Ignore request.arguments.body and use request.source_data instead
        request_body = request.source_data['request']
        
        if not all(key in request_body for key in EXPECTED_REQUEST_FIELDS):
            raise ValueError(f"Request body is missing expected fields: {EXPECTED_REQUEST_FIELDS}")
        
        response_handler = GenerationResponseHandlerFactory.create(request.request_type)

        request_info.timings.request_start = time.time()
        try:
            print("DEBUG: About to enter async with")
            async with self._async_client.stream(
                request.arguments.method or "POST",
                request_url,
                params=request.arguments.params,
                headers=request.arguments.headers,
                json=request_body,
            ) as stream:
                print("DEBUG: Inside async with, calling raise_for_status")
                stream.raise_for_status()
                print("DEBUG: Starting async for loop")
                end_reached = False
                async for chunk in stream.aiter_lines():
                    iter_time = time.time()

                    if request_info.timings.first_request_iteration is None:
                        request_info.timings.first_request_iteration = iter_time
                    request_info.timings.last_request_iteration = iter_time
                    request_info.timings.request_iterations += 1
                    
                    iterations = response_handler.add_streaming_line(chunk)
                    print("Iterations: ", iterations)
                    if iterations is None or iterations <= 0 or end_reached:
                        end_reached = end_reached or iterations is None
                        continue

                    if request_info.timings.first_token_iteration is None:
                        request_info.timings.first_token_iteration = iter_time
                        request_info.timings.token_iterations = 0

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += iterations
                print("DEBUG: Async for loop completed")
            print("DEBUG: Exited async with block")
        except Exception as e:
            print(f"DEBUG: Exception caught: {type(e).__name__}: {e}")
            raise
        finally:
            # Always set request_end timing, even if an exception occurred
            print("DEBUG: In finally block")
            request_info.timings.request_end = time.time()
        yield response_handler.compile_streaming(request), request_info
