

# Adology AWS Setup & Daily Operations Cookbook


This cookbook provides straightforward steps for setting up, deploying, and managing the Adology AWS-based application architecture.


## ✅ **Key Benefits**
- **Fast API Responses**: Simple queries return in **< 200ms**.
- **Efficient Async Processing**: Long tasks run in **background (SQS + workers)**.
- **Real-Time Updates**: WebSockets provide **instant status updates** to clients.
- **Scalability**: **Auto-scaling** for API servers, workers, and WebSockets.
- **Security**: Load Balancer (HTTPS, WSS) + VPC Peering + IAM Roles.




```
Client Frontend (Dashboards & JavaScript)
        │
        │ HTTPS API Calls & WebSockets (wss://)
        ▼
AWS Application Load Balancer (HTTPS/WSS)
        │
        ▼
Web/API Servers (EC2-based ECS Docker Containers)
 ├───────────────────┐
 │                   │
 │                   │
 ▼                   ▼
Immediate-mode       Long-running Async Tasks
(API Calls)          (Task Submission via SQS)
 │                   │
 │                   ▼
 │                AWS SQS Queue
 │                   │
 ▼                   ▼
Direct DB Calls      Backend Processing Workers
(MongoDB Atlas,      (EC2 Docker / Lambda Docker)
Postgres RDS)        │
 │                   │
 ▼                   ▼
 JSON Response       Sends Status Updates
 │                   │
 │                   ▼
 └──▶ Node.js WebSocket Server (EC2/Docker)
        │      ▲
        │      │
        ▼      │
Client WebSocket Connection (wss://)
   Receives real-time status updates
```
---
<div style="page-break-after: always;"></div>

## **Step-by-Step Walkthrough of the Adology System Architecture**

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
- ✅ **This completes the request instantly (~50-200ms response time).**

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

- ✅ The **dashboard updates instantly** without polling.

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

---
##   **Shared Managed Resources Overview**

The **shared managed resources** are foundational components that provide **reliable, scalable, and efficient** data storage, processing, and monitoring for the entire system. These services reduce operational overhead while ensuring security, availability, and performance.

---

### **1. Amazon SQS (Simple Queue Service) - Task Orchestration**
- **Purpose:** Provides **asynchronous messaging** to decouple Web/API requests from backend processing.
- **Use Case:** Handles long-running tasks that are too complex or time-consuming for immediate API responses.
- **Reliability:** **Guaranteed message delivery** with at-least-once processing and automatic retries.
- **Scalability:** Can handle **millions of messages per second**, ensuring the system adapts to traffic spikes.
- **Cost Efficiency:** Serverless, pay-per-use model—charges only for the number of messages sent and received.

---

### **2. MongoDB Atlas (Managed NoSQL Database)**
- **Purpose:** Stores flexible, high-performance JSON-like documents.
- **Use Case:** Fast retrieval of user dashboards, metrics, logs, and semi-structured data.
- **Deployment:** Hosted in AWS, with **VPC Peering** enabled for secure private access.
- **High Availability:** Multi-node replica sets ensure **zero downtime** in case of failure.

---

### **3. Amazon RDS (PostgreSQL - Managed Relational Database)**
- **Purpose:** Stores structured data, transactional records, and indexed relational data.
- **Use Case:** Ideal for dashboards, transactional workloads, and reporting queries.
- **Deployment:** **Multi-AZ (Availability Zone) configuration** for automatic failover.
- **Scalability:** **Automatic storage scaling** ensures smooth performance even with growing workloads.

---

### **4. Amazon S3 (Object Storage)**
- **Purpose:** Scalable storage for raw data, logs, and reports.
- **Use Case:** Stores large job results, logs, exported analytics, and user-generated content.
- **Redundancy:** **99.999999999% (11 9s) durability**, ensuring **data safety and availability**.

---

### **5. Amazon ECR (Docker Image Registry)**
- **Purpose:** Secure storage and versioning for Docker container images.
- **Use Case:** Hosts images for API servers, worker nodes, and WebSocket servers.
- **Efficiency:** Reduces bandwidth costs by keeping Docker images within AWS.

---

### **6. AWS CloudWatch (Monitoring & Alerts)**
- **Purpose:** Centralized monitoring for all AWS services.
- **Use Case:** Tracks CPU usage, memory, request rates, queue depths, and Lambda function invocations.
- **Automation:** Configured alarms trigger **auto-scaling** and notify operators of performance issues.


## 🛠️ Step-by-Step Setup Guide

### Step 1: Set Up Managed Databases

- **MongoDB Atlas**: Create a managed MongoDB cluster, enable AWS VPC Peering.
- **Amazon RDS PostgreSQL**: Use AWS RDS, Multi-AZ deployment recommended.

---

### Step 2: Set Up Amazon SQS Queues

- Create queues (`background-tasks`) for asynchronous communication between components.

---

### Step 2: Set Up Amazon ECR

- AWS Console → ECR → Create repository (e.g., `worker-containers`).
- Note the repository URL for future Docker push/pull commands.

---

### Step 3: Web/API ECS Cluster Setup

- AWS ECS → Create new ECS cluster (EC2-based).
- Use **t3.medium** EC2 instances (at least two) across two availability zones.
- Create and attach an **Application Load Balancer** (ALB) for HTTPS requests.

---

### Step 4: Install Docker on EC2 Instances (for ECS)

```bash
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -aG docker ec2-user
```

---

### Step 4: Deploy Docker Containers

Push your container image to ECR:

```bash
docker build -t worker-containers .
docker tag worker-containers:latest ACCOUNT_ID.dkr.ecr.region.amazonaws.com/worker-containers:latest
docker push ACCOUNT_ID.dkr.ecr.region.amazonaws.com/worker-containers:latest
```

ECS automatically pulls images from ECR.

---

### Step 5: Set Up AWS SQS

- AWS Console → SQS → "Create Queue".
- Name queue (e.g., `task-queue`), standard queue type.

---

### Step 5: Set Up Background Worker Processes (EC2/Lambda)

- **EC2 Workers**:
  - Launch EC2 instance, Docker installed.
  - Pull container images from ECR.
  - Workers poll AWS SQS for tasks.

- **AWS Lambda Overflow Workers**:
  - Create AWS Lambda from Docker image in ECR.
  - Set Lambda trigger directly from SQS.

---

### Step 6: Node.js WebSocket Server (Real-Time Updates)

Launch a dedicated EC2 instance (`t3.small`), and install Node.js WebSocket server:

```bash
sudo yum update -y
sudo yum install -y nodejs git
git clone YOUR_WEBSOCKET_REPO.git
cd YOUR_WEBSOCKET_REPO
npm install
node server.js
```

- Workers send updates via HTTP to this server.
- Server pushes updates via WebSocket to clients.

---

### Step 6b: WebSocket Frontend Client Example (JavaScript)

```javascript
const socket = new WebSocket('wss://your-domain.com');

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("Update:", data);
};
```

---

##  Daily Operations Checklist

- [ ] **AWS ECS Cluster Health Check**
- [ ] **Lambda function monitoring** (AWS Lambda console)
- [ ] **Check AWS SQS queue lengths**
- [ ] **MongoDB Atlas health check** (Atlas dashboard)
- [ ] **PostgreSQL RDS monitoring** (AWS RDS dashboard)
- [ ] **Node.js WebSocket Server status check** (EC2 health)

---

##  Routine Maintenance

**Weekly:**
- [ ] Update Docker images (ECR).
- [ ] Monitor CloudWatch alerts.
- [ ] Review AWS Billing to control costs.

---

## 💲 **Monthly Cost Estimation**

| Component                              | Cost/month (approx.) |
|----------------------------------------|----------------------|
| ECS Web/API EC2 (2 × t3.medium)        | $60                  |
| Application Load Balancer (ALB)        | $20                  |
| EC2 Background Workers (2 × t3.medium) | $60                  |
| AWS Lambda (overflow handling)         | $0–$50               |
| Amazon RDS PostgreSQL (db.t3.medium, Multi-AZ)| $120          |
| MongoDB Atlas (M10–M20)                | $55–$110             |
| Amazon S3 (Object Storage)             | $5                   |
| Amazon ECR (Docker Images)             | $5                   |
| WebSocket Server EC2 (t3.small)        | $15                  |
| AWS SQS                                | $5–$10               |
| AWS CloudWatch (basic monitoring)      | Included             |
| **Total Monthly Cost (corrected)**     | **~$345–$455**

## 🚀 Summary: Adology Architecture Benefits

- ✅ **Real-time updates**: WebSocket integration.
- ✅ **Immediate API responses**: EC2-based ECS.
- ✅ **Scalable asynchronous tasks**: EC2/Lambda via AWS SQS.
- ✅ **Managed database services**: MongoDB Atlas, PostgreSQL RDS.
- ✅ **Secure and highly available** (VPC peering, multi-AZ deployments).


