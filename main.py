from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import os
import time
import asyncio
import json
from openai import OpenAI

# Create FastAPI application
app = FastAPI()

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = "asst_LKmTIQgTYDE089Z2qzh6D30a"

# Pydantic models for validation
class UserQuery(BaseModel):
    question: str
    thread_id: Optional[str] = None

class AssistantResponse(BaseModel):
    response: str
    thread_id: str
    run_id: str

# Serve the home page with chat UI
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Helper function to stream assistant responses
async def stream_assistant_response(thread_id: str, run_id: str) -> AsyncGenerator[str, None]:
    # Initial delay to allow the run to start processing
    await asyncio.sleep(1)
    
    # Poll for status and stream results
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        if run_status.status == "completed":
            # Get all messages once the run is complete
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            
            # Find the latest assistant message
            latest_assistant_message = None
            for message in messages.data:
                if message.role == "assistant":
                    latest_assistant_message = message
                    break
            
            if latest_assistant_message:
                # Safely extract the content
                try:
                    if latest_assistant_message.content and len(latest_assistant_message.content) > 0:
                        content = latest_assistant_message.content[0].text.value
                        yield json.dumps({"type": "done", "content": content, "thread_id": thread_id, "run_id": run_id})
                    else:
                        yield json.dumps({"type": "done", "content": "No content in assistant response", "thread_id": thread_id, "run_id": run_id})
                except (IndexError, AttributeError) as e:
                    yield json.dumps({"type": "error", "content": f"Error extracting content: {str(e)}"})
            else:
                yield json.dumps({"type": "error", "content": "No assistant response found"})
            
            break
            
        elif run_status.status in ["failed", "cancelled", "expired"]:
            yield json.dumps({"type": "error", "content": f"Run failed with status: {run_status.status}"})
            break
            
        # Try to get any available messages for streaming while still running
        if run_status.status == "in_progress":
            messages = client.beta.threads.messages.list(thread_id=thread_id)
            
            for message in messages.data:
                if message.role == "assistant":
                    # Safely extract content for streaming
                    try:
                        if message.content and len(message.content) > 0:
                            content = message.content[0].text.value
                            yield json.dumps({"type": "update", "content": content})
                        else:
                            # Don't yield anything if there's no content yet
                            pass
                    except (IndexError, AttributeError):
                        # Skip if we can't extract content properly yet
                        pass
                    break
        
        # Wait before checking again
        await asyncio.sleep(1)

# Streaming API endpoint for asking questions
@app.post("/api/ask-stream")
async def ask_assistant_stream(query: UserQuery):
    try:
        # Create a new thread or use existing one
        thread_id = query.thread_id
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
        
        # Add a message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query.question
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        # Return a streaming response
        return StreamingResponse(
            stream_assistant_response(thread_id, run.id),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "content": str(e)})]),
            media_type="text/event-stream"
        )

# Keep the original non-streaming endpoint for backward compatibility
@app.post("/api/ask", response_model=AssistantResponse)
async def ask_assistant(query: UserQuery):
    # Original implementation
    try:
        # Create a new thread or use existing one
        thread_id = query.thread_id
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
        
        # Add a message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query.question
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )
        
        # Wait for completion
        import time
        start_time = time.time()
        timeout = 60  # 60 second timeout
        
        while True:
            if time.time() - start_time > timeout:
                raise HTTPException(status_code=504, detail="Assistant processing timed out")
            
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Assistant run failed with status: {run_status.status}"
                )
                
            time.sleep(1)
        
        # Get the assistant's response
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        
        if not messages.data:
            raise HTTPException(status_code=500, detail="No response from assistant")
        
        # Get the most recent assistant message
        for message in messages.data:
            if message.role == "assistant":
                response_text = message.content[0].text.value
                break
        else:
            raise HTTPException(status_code=500, detail="No assistant response found")
        
        return AssistantResponse(
            response=response_text,
            thread_id=thread_id,
            run_id=run.id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
