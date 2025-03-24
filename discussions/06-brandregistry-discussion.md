
## Active Brand Registry 

The singular Active Brand Registry Server  acts as a coordination service for distributed brand-level processing across multiple EC2 instances or containers. This service will manage state for each brand to ensure that:
	1.	Each brand is only processed once at a time, even when multiple workers consume from the same SQS queue.
	2.	All results for a given brand are batched and committed in one step, only after all URL-level jobs are completed.
	3.	No persistent storage is required — the server operates entirely in memory. If it fails, the worst-case scenario is re-processing some URLs, which our system can tolerate.

### Functional Responsibilities

The Active Brand Registry Server will:
	•	Accept a request to start tracking a brand’s batch processing.
	•	Receive processed results for each URL associated with a brand.
	•	Track progress toward completion of the batch.
	•	Signal when the full batch is complete and ready for commit.
	•	Allow exactly one worker to perform the final commit and trigger the post-commit AI phase.
	•	Clean up state when the commit is confirmed.

#### In-Memory State Per Brand

Each brand_id will be associated with:
	•	expected_count: Total number of URLs expected (set during Gather phase).
	•	completed_count: Number of processed URL results received.
	•	results: A list of result objects submitted by workers.
	•	commit_started: Boolean flag to ensure only one worker performs the commit.

#### Proposed HTTP API Contract

##### POST /start

Initialize tracking for a brand.

Request:
```json
{
  "brand": "nike",
  "expected_count": 27
}
```

Response:
200 OK

##### POST /submit

Submit a processed result for a URL. May trigger the commit if the batch is complete.

Request:
```json
{
  "brand": "nike",
  "result": {
    "url": "https://cdn.com/abc.jpg",
    "embedding": [...],
    "s3_path": "s3://bucket/abc.jpg"
  }
}
```
Response:
```json
{
  "should_commit": true  // or false
}
```

##### GET /results/{brand}

Returns all accumulated results for a brand. Used by the commit worker.

Response:
```json
[
  {
    "url": "...",
    "embedding": [...],
    "s3_path": "..."
  },
  ...
]
```

##### POST /commit-complete

Marks the brand as finished and cleans up all in-memory state.

Request:
```json
{ "brand": "nike" }
```
Response:
200 OK

#### Operational Notes
	•	The service will be single-instance and in-memory only.
	•	It will be called by multiple worker containers running on different machines.
	•	It should use internal locking to prevent race conditions across async requests.
	•	High availability or persistence is not required at this time.
 
----
<!--BREAK-->
