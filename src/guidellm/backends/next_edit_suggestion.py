
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Optional, Any

from guidellm.backends.response_handlers import GenerationResponseHandlerFactory, NESTextCompletionsResponseHandler
import httpx
from guidellm.backends.backend import Backend
from guidellm.schemas import (
    GenerationRequest,
    GenerationResponse,
    RequestInfo,
)

##### CONFIG #####
# Those are the information that needs to tweaked if the Next Edit Suggestion service interface changes
API_ROUTES = {
    "next_edit_suggestions": "/service/v5/generate/v1",
    "health": "/health",
}
OBLIGATORY_REQUEST_FIELDS = ['filepath', 'prefix', 'editable_region_prefix', 'editable_region_suffix', 'suffix', 'history']
OPTIONAL_REQUEST_FIELDS = ['max_length', 'temperature', 'seed', 'top_p', 'top_k']
END_OF_STREAM_LINE = "data: end"
##################


@Backend.register("next_edit_suggestion")
class NextEditSuggestionBackend(Backend):
    """
    HTTP backend for Next Edit Suggestion service.
    
    The intention is to keep this backend minimal.
    
    Example CLI command to run the benchmark:
    guidellm benchmark   --target "http://0.0.0.0:5000"  --profile constant  --rate 1  --data nes_dumps_2000.json --backend next_edit_suggestion --request-type text_completions --processor next_edit_suggestion_column_mapper
    
    Note: --data must point to the **our** NES dumps file from jetbrains-ai-qa. 
    
    :param target: The base URL of the Next Edit Suggestion service (e.g. 'http://0.0.0.0:5000')
    :param model: The model to use for the Next Edit Suggestion service (currently not used)
    :param api_routes: The API routes to use for the Next Edit Suggestion service
    :param timeout: The timeout for the HTTP requests to the Next Edit Suggestion service
    """
    def __init__(self, target: str, model: Optional[str] = None, api_routes: dict[str, str] = API_ROUTES, timeout: float = 60.0):
        super().__init__(type_="next_edit_suggestion")
        self.api_routes = api_routes
        self.target = target
        self.timeout = timeout
        
        # Runtime state
        self._in_process = False
        self._async_client: httpx.AsyncClient | None = None
        
    async def process_startup(self):
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")
        
        self._async_client = httpx.AsyncClient(timeout=self.timeout)
        self._in_process = True

    async def process_shutdown(self):
        """
        Clean up HTTP client and backend resources.

        :raises RuntimeError: If backend was not properly initialized
        :raises httpx.RequestError: If HTTP client cannot be closed
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        await self._async_client.aclose()  # type: ignore [union-attr]
        self._async_client = None
        self._in_process = False
    
    @property
    def info(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "type": self.type_,
            "api_routes": self.api_routes,
            "timeout": self.timeout,
        }

    async def validate(self):
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        try:
            response = await self._async_client.get(self.target + self.api_routes['health'])
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                "Backend validation request failed. Could not connect to the server "
                "or validate the backend configuration."
            ) from e
        
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
        
        Handles exclusively the streaming response from the Next Edit Suggestion service.
        
        :param request: Generation request with content and parameters
        :param request_info: Request tracking info updated with timing metadata
        :param history: Conversation history (currently not supported)
        :yields: Tuples of (response, updated_request_info) as generation progresses
        """
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")
        
        if history is not None:
            raise NotImplementedError("Multi-turn requests not yet supported")
        
        if (request_path := self.api_routes.get(request.request_type)) is None:
            raise ValueError(f"Unsupported request type '{request.request_type}'. Supported types: {list(self.api_routes.keys())}")
        
        request_url = f"{self.target}{request_path}"
        print("Body:", request.arguments.body)
        
        self._validate_request_body(request.arguments.body)
        
        response_handler: NESTextCompletionsResponseHandler = GenerationResponseHandlerFactory.create(request.request_type)

        request_info.timings.request_start = time.time()
        
        # Here is where we actually collect the streaming response from the Next Edit Suggestion service
        try:
            async with self._async_client.stream(
                request.arguments.method or "POST",
                request_url,
                params=request.arguments.params,
                headers=request.arguments.headers,
                json=request.arguments.body,
            ) as stream:
                stream.raise_for_status()
                end_reached = False
                async for chunk in stream.aiter_lines():
                    iter_time = time.time()

                    if request_info.timings.first_request_iteration is None:
                        request_info.timings.first_request_iteration = iter_time
                    request_info.timings.last_request_iteration = iter_time
                    request_info.timings.request_iterations += 1
                    
                    iterations = response_handler.add_streaming_line(chunk, end_of_stream_line=END_OF_STREAM_LINE)
                    if iterations is None or iterations <= 0 or end_reached:
                        end_reached = end_reached or iterations is None
                        continue

                    if request_info.timings.first_token_iteration is None:
                        request_info.timings.first_token_iteration = iter_time
                        request_info.timings.token_iterations = 0

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += iterations
        except Exception as e:
            raise RuntimeError(f"Error processing streaming response from Next Edit Suggestion service: {e}") from e
        finally:
            request_info.timings.request_end = time.time()
        yield response_handler.compile_streaming(request), request_info
        
    def _validate_request_body(self, request_body: dict):
        """
        Validate the request body according to the NES service interface.
        
        :param request_body: The request body to validate
        :raises ValueError: If the request body is missing expected fields or contains superfluous fields
        """
        if not all(key in request_body for key in OBLIGATORY_REQUEST_FIELDS):
            raise ValueError(f"Request body is missing expected fields: {OBLIGATORY_REQUEST_FIELDS}. Got: {request_body.keys()}")
        superfluous_fields = [key for key in request_body.keys() if key not in OBLIGATORY_REQUEST_FIELDS + OPTIONAL_REQUEST_FIELDS]
        if superfluous_fields:
            raise ValueError(f"Request body contains superfluous fields: {superfluous_fields}. Got: {request_body.keys()}")