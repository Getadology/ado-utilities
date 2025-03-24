## 3-Phase Pipeline

Below is a text-based ASCII-style diagram of the architecture, which you can copy into a Markdown file. It shows the flow from the frontend making an API or WebSocket request, through the AWS services, and finally back via real-time updates.
```

                                 ┌─────────────────────┐
                                 │  Frontend (Client)  │
                                 │ (JS Dashboard)      │
                                 └─────────┬───────────┘
                                           │
                                           ▼
                          ┌────────────────────────────────┐
                          │ AWS Application Load Balancer  │
                          │ (ALB)                          │
                          └─────────┬──────────────────────┘
             HTTPS/API Requests     │                  WebSocket Requests
  ┌─────────────────────────────────┘                            └────────────────────────────────────┐
  │                                                                                                   ▼
  │                                          ┌──────────────────────────────────────────┐    ┌───────────────────────┐
  │                                          │ ECS Web/API Servers (EC2-based Docker)   │    │ WebSocket Server (EC2)│
  │                                          │  - Immediate-mode requests               │    │   (broadcast updates) │
  │                                          │  - Long-running tasks → SQS              │    └─────────────┬─────────┘
  │                                          └─────────┬─────────────┬──────────────────┘                │
  │                                                    │             │                                   │
  │                                                    │ (Immediate) │ (Long-running)                    │
  │                                                    ▼             ▼                                   │
  │                                       ┌────────────────────┐   ┌──────────────────┐                  │
  │                                       │  MongoDB / RDS     │   │     AWS SQS      │                  │
  │                                       │  (Databases)       │   │ (Queue for tasks)│                  │
  │                                       └────────────────────┘   └──────────────────┘                  │
  │                                                                   │                                  │
  │                                                                   ▼                                  │
  │                                                      ┌──────────────────────────┐                    │
  │                                                      │   Backend Workers       │                     │
  │                                                      │ (ECS Docker / Lambda)   │                     │
  │                                                      └─────────────┬───────────┘                     │
  │                                                                    │ (Processed results)             │
  │                                                                    ▼                                 │
  └─────────────────────────────────────────────────────────────────────┴────────────────────────────────┘
                                                                       │
                                                                       ▼
                                                       Frontend receives real-time
                                                         status updates over WebSocket
```

#### Explanation of the Flow

  1.  Frontend (Client)
  •  A JavaScript dashboard in the browser that either makes API (HTTPS) requests or maintains a WebSocket connection for real-time updates.
  2.  AWS Application Load Balancer (ALB)
  •  Routes incoming traffic:
  •  HTTPS requests to ECS Web/API Servers.
  •  WebSocket requests to the WebSocket Server.
  3.  ECS Web/API Servers (EC2-based Docker)
  •  Immediate-mode requests (fast queries) go directly to MongoDB / RDS and return results right away.
  •  Long-running requests enqueue jobs to AWS SQS, then immediately respond with a “task submitted” message.
  4.  Databases (MongoDB / RDS)
  •  Used for quick lookups, analytics, or any data required by the immediate-mode API calls.
  5.  AWS SQS (Queue for long-running tasks)
  •  Stores and manages background jobs that take longer to complete.
  6.  Backend Workers (ECS or Lambda)
  •  Pull tasks from SQS and process them (e.g., data aggregation, AI/ML tasks).
  •  After processing, they send status/results to the WebSocket Server.
  7.  WebSocket Server (EC2)
  •  Receives status updates from the backend workers.
  •  Broadcasts these updates over the WebSocket connection to any subscribed frontend clients.
  8.  Real-Time Frontend Updates
  •  The JavaScript dashboard receives these updates instantly, allowing the UI to refresh without polling the server.


## Phase by Phase

### 1. Gather Phase
  •  Queue: brand-gather-queue
  •  Input: { "brand": "nike" }
  •  Responsibility:
  •  Call external service to get list of URLs + metadata.
  •  For each URL, enqueue a message to url-process-queue.

### 2. Acquire & Commit Phase
  •  Queue: url-process-queue
  •  Input:
```json
{
  "brand": "nike",
  "url": "https://cdn.com/image.jpg",
  "metadata": { ... }
}
```

  •  Responsibility (per URL):
  •  Download content to EC2.
  •  Upload to S3.
  •  Compute embedding.
  •  Store result in local buffer for that brand.
  •  Responsibility (per Brand):
  •  Use local tracking (e.g. a counter or barrier) to detect when all URLs for a brand are finished.
  •  When complete:
  •  Insert the full batch (all URLs + embeddings) to the DB in one transaction.
  •  Then enqueue a message to the AI phase queue.

### 3. Post-Commit AI Phase
  •  Queue: brand-ai-phase-queue
  •  Input: { "brand": "nike" }
  •  Responsibility:
  •  Perform asynchronous brand-wide AI processing.

#### Notes
  •  Each worker processes URL-level work independently.
  •  Active Brand Registry tracks:
   - Total expected URLs per brand.
   - When all URLs for a brand are done.
    - Once the last one is processed → trigger commit inline.

 ----
 
<!--BREAK-->
