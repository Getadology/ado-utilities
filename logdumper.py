"""
"To create a Python utility that fetches exceptional log entries for a specific Lambda function from CloudWatch, you can use boto3 to interact with AWS services. Here’s a utility to get you started:
"""

import boto3
import re
from datetime import datetime, timedelta

def get_lambda_log_group(function_name):
    return f'/aws/lambda/{function_name}'

def fetch_exception_logs(function_name, hours=1):
    logs_client = boto3.client('logs')
    log_group_name = get_lambda_log_group(function_name)

    # Calculate the time window
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    # Search for exception logs
    response = logs_client.filter_log_events(
        logGroupName=log_group_name,
        startTime=int(start_time.timestamp() * 1000),
        endTime=int(end_time.timestamp() * 1000),
        filterPattern='?Exception ?ERROR ? Error'
    )

    for event in response['events']:
        timestamp = datetime.fromtimestamp( event['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {event['message']}")

# Example usage
if __name__ == '__main__':
    fetch_exception_logs('your-lambda-function-name', hours=2)

"""
To use this utility:
  1.  Replace 'your-lambda-function-name' with the name of your Lambda function.
  2.  Install boto3 with pip install boto3.
  3.  Configure your AWS credentials.

This utility will search for logs containing keywords like “Exception” or “ERROR” within the specified time window. You can adjust the filterPattern and the time window as needed. If you have any more specific requirements, feel free to let me know!
"""
