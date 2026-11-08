"""One error format for every failure: {"error": {"code", "message", "request_id"}}.

Clients get a stable machine-readable code and a request id to quote; stack traces and
internal details stay in the server log.
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


_HTTP_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
               405: "method_not_allowed", 429: "rate_limited", 503: "unavailable"}


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(request: Request, status: int, code: str, message: str,
                   headers: dict[str, str] | None = None, **extra) -> JSONResponse:
    rid = request_id_of(request)
    body = {"error": {"code": code, "message": message, "request_id": rid, **extra}}
    return JSONResponse(body, status_code=status, headers={**(headers or {}), **({"X-Request-ID": rid} if rid else {})})


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(request, exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return error_response(request, exc.status_code, _HTTP_CODES.get(exc.status_code, "http_error"),
                              message, getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Field location and reason only: the rejected input is not echoed back.
        details = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
                   for e in exc.errors()]
        return error_response(request, 422, "invalid_request", "The request is invalid.", details=details)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.opt(exception=exc).error(f"Unhandled error (request_id={request_id_of(request)})")
        return error_response(request, 500, "internal_error", "Something went wrong on our side.")
