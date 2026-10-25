
import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse


import hashlib
import json
import uuid

from storage import (
    canonical_json,
    complete_message_claim,
    find_message,
    get_task,
    list_tasks as storage_list_tasks,
    save_new_task,
    update_task,
    reserve_message,
)
from invoice_agent import decide_batch


def find_data_part(message):
    for part in message.get("parts", []):
        if isinstance(part, dict) and isinstance(part.get("data"), dict):
            return part["data"]
    return None


def get_proposals(task):
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            data = part.get("data")
            if isinstance(data, dict) and isinstance(data.get("proposals"), list):
                return data["proposals"]
    return None



app = FastAPI(title="GA3 Invoice Action Agent")

BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "http://127.0.0.1:8000/a2a/"
).rstrip("/") + "/"

API_TOKEN = os.getenv("A2A_BEARER_TOKEN", "dev-token-change-me")

PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
BATCH = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS = "application/vnd.ga5.invoice-action-results+json"


@app.get("/.well-known/agent-card.json")
async def agent_card():
    return {
        "name": "Invoice Action Agent",
        "description": (
            "An evidence-based invoice agent that proposes "
            "safe business actions and records approved executions."
        ),
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json",
            "application/vnd.ga5.invoice-action-results+json"
        ],
        "defaultOutputModes": [PROPOSALS, RECEIPTS],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Reviews invoice evidence and proposes or "
                    "executes only explicitly approved actions."
                ),
                "tags": [
                    "invoices",
                    "evidence",
                    "approval",
                    "idempotency"
                ]
            }
        ]
    }


async def validate_request(request: Request):
    authorization = request.headers.get("authorization", "")

    # Require a well-formed Bearer token.
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization[len("Bearer "):]

    if not token or token.strip() != token or " " in token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # The token identifies the principal; different tokens are
    # separate users for task isolation.
    if request.headers.get("a2a-version") != "1.0":
        raise HTTPException(status_code=400, detail="Invalid protocol version")

    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if content_type.split(";")[0].strip() != "application/a2a+json":
            raise HTTPException(
                status_code=415,
                detail="Unsupported media type"
            )

    return token

def extract_message(body):
    message = body.get("message")
    if not isinstance(message, dict):
        raise HTTPException(400, detail="Missing message object")

    if not isinstance(message.get("messageId"), str) or not message["messageId"]:
        raise HTTPException(400, detail="Missing messageId")

    if message.get("role") != "ROLE_USER":
        raise HTTPException(400, detail="Expected ROLE_USER")

    parts = message.get("parts", [])
    batch = None

    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("data"), dict):
            candidate = part["data"]
            if isinstance(candidate.get("packages"), list):
                batch = candidate
                break

    if batch is None:
        raise HTTPException(400, detail="Missing invoice batch data")

    if not isinstance(batch.get("batchId"), str) or not batch["batchId"]:
        raise HTTPException(400, detail="Missing batchId")

    return message, batch


def make_proposal_task(message, proposal_data):
    task_id = str(uuid.uuid4())
    context_id = message.get("contextId") or str(uuid.uuid4())

    return {
        "id": task_id,
        "contextId": context_id,
        "batchId": proposal_data.get("batchId"),
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED"
        },
        "artifacts": [
            {
                "artifactId": str(uuid.uuid4()),
                "name": "invoice_action_proposals",
                "parts": [
                    {
                        "kind": "data",
                        "data": proposal_data
                    }
                ]
            }
        ],
        "history": [message]
    }


@app.get("/")
async def root():
    return {"status": "Invoice agent is running"}




@app.post("/a2a/message:send")
async def send_message(request: Request):
    await validate_request(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(400, detail="Expected JSON object")

    message = body.get("message")
    if not isinstance(message, dict):
        raise HTTPException(400, detail="Missing message")

    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(400, detail="Missing messageId")

    if message.get("role") != "ROLE_USER":
        raise HTTPException(400, detail="Expected ROLE_USER")

    principal = request.headers["authorization"][len("Bearer "):]
    message_hash = hashlib.sha256(
        canonical_json(message).encode("utf-8")
    ).hexdigest()

    # A replay of an initial request must not call Gemini again.
    
    reservation = reserve_message(
        principal, message_id, message_hash
    )

    if reservation["status"] == "conflict":
        return JSONResponse(
            status_code=409,
            content={"code": "IDEMPOTENCY_CONFLICT"}
        )

    if reservation["status"] == "completed":
        return reservation["task"]

    if reservation["status"] == "processing":
        return JSONResponse(
            status_code=409,
            content={"code": "REQUEST_IN_PROGRESS"}
        )


    data = find_data_part(message)
    if data is None:
        raise HTTPException(400, detail="Missing data part")

    # Initial invoice batch
    if isinstance(data.get("packages"), list):
        if not isinstance(data.get("batchId"), str):
            raise HTTPException(400, detail="Missing batchId")

        try:
            proposal_data = decide_batch(data)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, detail=str(exc))

        task = make_proposal_task(message, proposal_data)

        try:
            
            saved_task = save_new_task(
                task,
                principal,
                message_id,
                message_hash,
                data["batchId"]
            )

            complete_message_claim(
                principal, message_id, message_hash, saved_task
            )

            return saved_task

        except ValueError as exc:
            if str(exc) == "IDEMPOTENCY_CONFLICT":
                return JSONResponse(
                    status_code=409,
                    content={"code": "IDEMPOTENCY_CONFLICT"}
                )
            raise

    # Result continuation
    results = data.get("results")
    task_id = message.get("taskId") or body.get("taskId")
    context_id = message.get("contextId") or body.get("contextId")

    if not isinstance(results, list) or not task_id:
        raise HTTPException(400, detail="Invalid result continuation")

    task = get_task(task_id, principal)
    if task is None:
        raise HTTPException(404, detail="Task not found")

    if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
        raise HTTPException(409, detail="Task is already terminal")

    if context_id != task["contextId"]:
        raise HTTPException(409, detail="Context mismatch")

    if data.get("batchId") != task.get("batchId"):
        raise HTTPException(409, detail="Batch mismatch")

    proposals = get_proposals(task)
    if proposals is None:
        raise HTTPException(409, detail="Stored proposals missing")

    proposal_map = {p["packageId"]: p for p in proposals}
    seen = set()
    accepted = []

    for result in results:
        if not isinstance(result, dict):
            raise HTTPException(400, detail="Invalid result entry")

        package_id = result.get("packageId")
        proposal = proposal_map.get(package_id)

        if proposal is None or package_id in seen:
            raise HTTPException(409, detail="Unknown or duplicate package")

        seen.add(package_id)

        if (
            result.get("actionId") != proposal["actionId"]
            or result.get("action") != proposal["action"]
        ):
            raise HTTPException(409, detail="Proposal mismatch")

        outcome = result.get("outcome")
        if outcome not in ("ACCEPTED", "REJECTED"):
            raise HTTPException(400, detail="Invalid outcome")

        if outcome == "ACCEPTED":
            nonce = result.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                raise HTTPException(400, detail="Missing receiptNonce")

            accepted.append({
                "packageId": package_id,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": nonce
            })

    if seen != set(proposal_map):
        raise HTTPException(409, detail="Results must cover every proposal")

    # Record the continuation and the accepted receipts.
    task["history"].append(message)
    task["artifacts"].append({
        "artifactId": str(uuid.uuid4()),
        "name": "invoice_action_receipts",
        "parts": [{
            "kind": "data",
            "data": {
                "batchId": data["batchId"],
                "receipts": accepted
            }
        }]
    })
    task["status"]["state"] = "TASK_STATE_COMPLETED"

    
    try:
        update_task(task, principal)
        complete_message_claim(
            principal, message_id, message_hash, task
        )
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return task




@app.get("/a2a/tasks")
async def list_tasks(request: Request):
    await validate_request(request)
    principal = request.headers["authorization"][len("Bearer "):]
    return {"tasks": storage_list_tasks(principal)}


@app.get("/a2a/tasks/{task_id}")
async def get_task_endpoint(task_id: str, request: Request):
    await validate_request(request)
    principal = request.headers["authorization"][len("Bearer "):]

    task = get_task(task_id, principal)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return task



@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    await validate_request(request)
    principal = request.headers["authorization"][len("Bearer "):]

    task = get_task(task_id, principal)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
        raise HTTPException(status_code=409, detail="Task is already terminal")

    task["status"]["state"] = "TASK_STATE_CANCELED"

    try:
        update_task(task, principal)
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return task

