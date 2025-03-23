# Do Later

## Mongo Must Do
### Monitor Screens
### Modularize and Rework All Updates
#### Add Transactions
#### Add Locks?
Hopefully won't be needed
#### Add Timings
## SQS Must Do
### SQS Processing Framework
### Monitor Screen
### Associate One Lambda Per Queue
It should also be possible to associate zero, in which case the queue can feed a local ec2 process.
### Limit Max Lambda Instances Per Queue
Important to limit the fanout, particulary when locking is introduced.
### Specify Message PayLoad Details Per Queue
For pedagogical purposes.
### Make Batching Optional
Might be unnecessary in some cases where we are certain the lambda will complete in under 15 mins.
## SQS Lamdbas
At this point all lambdas are sqs driven. 
### Dictionary of Lambdas
For pedagogical purposes.
### Performance Measurement Harness
 At the outer level all lambdas look pretty much the same so we can grab some measurements before and after we run the lambas.



