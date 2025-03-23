# Workflow  
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
  ┌───────────────────────────────────────────┐                  └────────────────────────────────────┐
  │                                           ▼                                                       ▼
  │                                          ┌──────────────────────────────────────────┐    ┌────────────────────────┐
  │                                          │ ECS Web/API Servers (EC2-based Docker)   │    │ WebSocket Server (EC2) │
  │                                          │  - Immediate-mode requests               │    │   (broadcast updates)  │
  │                                          │  - Long-running tasks → SQS              │    └─────────────┬──────────┘
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
  │                                                      ┌─────────────────────────┐                     │
  │                                                      │   Backend Workers       │                     │
  │                                                      │ (ECS Docker / Lambda)   │                     │
  │                                                      └─────────────┬───────────┘                     │
  │                                                                    │ (Processed results)             │
  │                                                                    ▼                                 │
  └────────────────────────────────────────────────────────────────────┴─────────────────────────────────┘
                                                                       │
                                                                       ▼
                                                       Frontend receives real-time
                                                       status updates over WebSocket


```
 
 ## **Step-by-Step Walkthrough of the Adology System Requests**

This section provides a **step-by-step walkthrough** of the architecture, describing how **immediate-mode API calls** and **long-running async tasks** are handled, including **real-time WebSocket updates**.

---

### **Step 1: Frontend Client Sends a Request**
- The **frontend dashboard (JavaScript client)** needs data.
- It can make a request in **two ways**:
  1. **Immediate-mode API request** (for fast database queries).
  2. **Asynchronous task submission** (for long-running background jobs).

---

### **Step 2: Request Reaches the Load Balancer**
- The **AWS Application Load Balancer (ALB)** receives the request.
- ALB is responsible for:
  - **HTTPS API requests** → Routed to ECS Web/API servers.
  - **WebSockets (wss://) connections** → Routed to the WebSocket server.

---

### **Step 3: Request is Handled by Web/API Servers**
- The request reaches **EC2-based ECS Docker containers** (API servers).
- The API server checks the **type of request**:
  1. **Immediate-mode API Call** → Queries the database and returns a JSON response.
  2. **Long-running Request** → Sends a task to **AWS SQS** for background processing.

---

### **Step 4A: Immediate API Requests (Handled Instantly)**
- If the request is **simple (e.g., fetching dashboard data)**:
  - The API server **queries MongoDB Atlas or Amazon RDS (PostgreSQL)** directly.
  - A **JSON response** is immediately sent back to the frontend.
  - Example:
    ```json
    {
      "user_count": 12000,
      "active_sessions": 320,
      "avg_response_time_ms": 45
    }
    ``` 

---

### **Step 4B: Long-Running Requests (Sent to SQS)**
- If the request **requires extensive computation** (e.g., AI processing, data aggregation):
  - The API server **sends the request** to **AWS SQS** for background processing.
  - The API **immediately returns a confirmation**:
    ```json
    {
      "request_id": "abc123",
      "status": "processing",
      "message": "Your task has been queued."
    }
    ```

---

### **Step 5: Backend Workers Process Async Tasks**
- Backend worker processes handle tasks **asynchronously**:
  1. **EC2 Docker Workers** (for steady workloads).
  2. **AWS Lambda Functions** (for overflow handling).
- Workers **pull tasks from AWS SQS**, process them, and prepare results.

---

### **Step 6: Workers Send Status Updates to WebSocket Server**
- Once a task is processed, the backend worker **sends a status update** to the **WebSocket server**.
- The WebSocket server is a **Node.js application** running on EC2, designed to broadcast real-time messages.

- Example **Python code (backend worker sends update):**
  ```python
  import requests

  def send_update_to_websocket(status):
      endpoint = "http://websocket-server-ip:8080"
      payload = {"request_id": "abc123", "status": "completed"}

      response = requests.post(f"{endpoint}/notify", json=payload)
      print("Broadcast response:", response.text)
  ```

- The WebSocket server **receives the update** and sends it to **all connected frontend clients**.

---

### **Step 7: WebSocket Server Pushes Updates to Clients**
- The frontend **maintains an active WebSocket connection** (`wss://`).
- When an update arrives, the **WebSocket server pushes it to clients** in real-time.

- **Example WebSocket broadcast message (JSON format):**
  ```json
  {
    "request_id": "abc123",
    "status": "completed",
    "result": "Your task has finished processing!"
  }
  ```

---

### **Step 8: Frontend Receives Updates in Real-Time**
- The **JavaScript client listens for updates** via WebSockets.
- **Example WebSocket Client Code (JavaScript):**
  ```javascript
  const socket = new WebSocket('wss://your-websocket-domain.com');

  socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received update:", data);
  };
  ```

-  The **dashboard updates instantly** without polling.

---

### **Step 9: Summary of Workflow**

| **Step** | **Process** |
|----------|------------|
| **1** | Frontend makes a request (API call or async task). |
| **2** | AWS ALB routes the request to API servers. |
| **3** | API server processes **immediate requests** or queues **async tasks**. |
| **4A** | **Immediate response**: API returns data from MongoDB/PostgreSQL. |
| **4B** | **Long-running task**: API queues task in AWS SQS. |
| **5** | Backend workers (EC2/Lambda) **process tasks**. |
| **6** | Workers send **status updates** to WebSocket server. |
| **7** | WebSocket server **broadcasts updates** to clients. |
| **8** | Frontend **receives real-time updates** via WebSockets. |

====
