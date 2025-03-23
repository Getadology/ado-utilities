#
"""
//  sqsdump.py
//

To build a Python utility for listing SQS queues along with their attributes, you can use the boto3 library, which is the AWS SDK for Python. This utility will fetch the list of queues and their details, including their message counts. Here’s a basic implementation:
"""
import boto3

def list_sqs_queues(queue_names):
    # Initialize a session using Amazon SQS
    sqs = boto3.client('sqs')

    # Dictionary to hold queue details
    queue_details = {}

    for queue_name in queue_names:
        # Get the queue URL
        response = sqs.get_queue_url(QueueName=queue_name)
        queue_url = response['QueueUrl']

        # Get the queue attributes
        response = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['All']
        )

        # Store the relevant details
        attributes = response['Attributes']
        queue_details[queue_name] = {
            'ApproximateNumberOfMessages': attributes.get('ApproximateNumberOfMessages', '0'),
            'ApproximateNumberOfMessagesNotVisible': attributes.get('ApproximateNumberOfMessagesNotVisible', '0'),
            # Add other attributes if needed
        }

    # Print the details
    for queue_name, details in queue_details.items():
        print(f"Queue: {queue_name}")
        for key, value in details.items():
            print(f"  {key}: {value}")
        print()

# Example usage
if __name__ == '__main__':
    queue_names = ['my-queue-1', 'my-queue-2']
    list_sqs_queues(queue_names)
"""
To use this code:
	1.	Install boto3 with pip install boto3.
	2.	Configure your AWS credentials for boto3 using the AWS CLI or environment variables.

You can extend this utility to fetch and display additional attributes if needed.
"""
