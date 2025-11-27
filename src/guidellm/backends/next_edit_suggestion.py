
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Optional, Any

import httpx

from guidellm.backends.backend import Backend
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
    UsageMetrics,
)

API_ROUTES = {
    "text_completions": "/service/v5/generate/v1",
    "health": "/health",
}




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
        
        print("Request URL: ", request_url)
        print("Request Body: ", request.arguments.body)
        print("Request Params: ", request.arguments.params)
        print("Request Headers: ", request.arguments.headers)
        print("Request Method: ", request.arguments.method)
        print("Request Type: ", request.request_type)
        print("Request ID: ", request.request_id)
        print("Request Args: ", request.arguments.model_dump())
        print("Request Info: ", request_info)
        print("History: ", history)

        # Make the API request
        request_info.timings.request_start = time.time()
        try:
            # Create a copy of the body and remove stream-related keys (hack)
            request_body = dict(request.arguments.body) if request.arguments.body else {}
            request_body.pop('stream', None)
            request_body.pop('stream_options', None)
            
            print("Request method: ", request.arguments.method)
            print("Request url: ", request_url)
            print("Request params: ", request.arguments.params)
            print("Request headers: ", request.arguments.headers)
            print("Request body: ", request_body)
            
            response = await self._async_client.request(
                request.arguments.method or "POST",
                request_url,
                params=request.arguments.params,
                headers=request.arguments.headers,
                json=request_body,
            )
            request_info.timings.request_end = time.time()
            
            # Log response details for debugging
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            response.raise_for_status()
            
            # Parse response and yield
            try:
                data = response.json()
            except Exception as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Response text: {response.text[:500]}")  # First 500 chars
                raise
                
        except httpx.HTTPStatusError as e:
            request_info.timings.request_end = time.time()
            print(f"HTTP Error {e.response.status_code}: {e.response.text[:500]}")
            raise
        except httpx.RequestError as e:
            request_info.timings.request_end = time.time()
            print(f"Request Error: {e}")
            raise
        except Exception as e:
            request_info.timings.request_end = time.time()
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise
        
        # Extract text from response (adjust based on actual API format)
        text = data.get("text") or data.get("content") or ""
        
        # Extract usage metrics if available
        usage = data.get("usage", {})
        input_metrics = UsageMetrics(
            text_tokens=usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
        )
        output_metrics = UsageMetrics(
            text_tokens=usage.get("completion_tokens") or usage.get("output_tokens") or 0,
            text_words=len(text.split()) if text else 0,
            text_characters=len(text) if text else 0,
        )
        
        generation_response = GenerationResponse(
            request_id=request.request_id,
            request_args=str(
                request.arguments.model_dump() if request.arguments else None
            ),
            response_id=data.get("id"),
            text=text,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )
        
        yield generation_response, request_info