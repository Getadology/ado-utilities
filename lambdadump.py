#
//  lambdadump.py
//  ado-utilities
//
//  Created by bill donner on 3/1/25.
//

"""
To get more detailed information about each Lambda function, you can use the boto3 library to fetch various metrics and configurations. However, some of the details, like CPU time and specific SQS connections, are not directly available from Lambda’s basic API. Here’s how you can get started:
	1.	Lambda Function Details: You can get the function’s configuration, code size, and runtime metrics.
	2.	Execution Metrics: You can access CloudWatch logs to get insights into execution times, errors, and invocations.
	3.	SQS Queue Connections: If your Lambda is triggered by SQS, you’ll need to check the event source mappings.

Here’s an example utility to start with:
"""

import boto3
import json

def list_lambda_details():
    lambda_client = boto3.client('lambda')
    cloudwatch_client = boto3.client('cloudwatch')

    function_details = {}

    paginator = lambda_client.get_paginator('list_functions')
    for page in paginator.paginate():
        for function in page['Functions']:
            function_name = function['FunctionName']
            
            # Get basic configuration
            function_details[function_name] = {
                'Runtime': function['Runtime'],
                'Handler': function['Handler'],
                'CodeSize': function['CodeSize'],
                'LastModified': function['LastModified'],
            }
            
            # Get CloudWatch metrics (e.g., Invocations, Duration)
            metrics = cloudwatch_client.get_metric_statistics(
                Period=300,
                StartTime=datetime.utcnow() - timedelta(days=1),
                EndTime=datetime.utcnow(),
                MetricName='Invocations',
                Namespace='AWS/Lambda',
                Statistics=['Sum'],
                Dimensions=[{'Name': 'FunctionName', 'Value': function_name}]
            )
            
            if metrics['Datapoints']:
                function_details[function_name]['Invocations'] = metrics['Datapoints'][0]['Sum']

    # Print the details
    for function_name, details in function_details.items():
        print(f"Function: {function_name}")
        print(json.dumps(details, indent=4))
        print()

if __name__ == '__main__':
    list_lambda_details()

"""
To use this utility:
	1.	Install boto3 with pip install boto3.
	2.	Configure your AWS credentials.

This example fetches function configurations and basic CloudWatch metrics (like invocations). For detailed metrics like CPU usage and I/O operations, you’d typically need to integrate CloudWatch Logs and set up custom logging within your Lambda functions. To identify SQS connections, you would need to check the event source mappings in Lambda, which can be done using list_event_source_mappings API.

Would you like guidance on fetching event source mappings to identify which Lambdas are connected to SQS queues?
"""
