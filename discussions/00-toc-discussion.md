# Adology Background

Adology is a intelligent service that analyzes digital advertising by applying knowledge processing to evaluate content(video, image, and text) as it is ingested into a permanent Adology repository.

At its core, Adology is not just a passive ad-tracking tool. Instead, it functions as an AI-powered intelligence engine that captures:

- The raw reality of ad creatives (metadata, images, videos, and text)
- The structured interpretation of these creatives (AI-generated labels, scores, and embeddings)
- The higher-level knowledge extracted from AI insights (brand-wide trends, comparative analysis, and strategic reports)

<!--TOC-->

## Table of Contents
- [Overview](#overview)
- [AWS Environment](#aws-environment)
- [Workflow](#workflow)
- [Logging and Metrics](#logging-and-metrics)
- [Containers and Concurrency](#containers-and-concurrency)
- [3-Phase Pipeline](#3-phase-pipeline)
- [Active Brand Registry](#active-brand-registry)
- [Ops Dashboard](#ops-dashboard)
- [Development and Testing](#development-and-testing)

## Overview 



Adology’s architecture is designed to systematically store, analyze, and generate insights from advertising data. The system must support highly structured AI-driven analysis while also maintaining efficient retrieval of brand, ad, and intelligence reports—all within a cost-effective, scalable database structure.


Adology's data architecture supports multiple users within customer accounts, each managing multiple *brandspaces*. Each brandspace focuses on a primary brand, tracking competitor brands and followed brands to define the scope of intelligence gathering. A user can switch brandspaces for one customer organization at any time but requires a unique email and separate login to support multiple organizations.

Adology largely operates as a conventional, Internet-accessible database and content management service built on top of well-known data stores and access methods, including AWS, EC2, S3, SQS, Lambda, Postgres, Mongo, and Python. Its key value lies in intelligent, contextual analysis and notifications for advertising-based videos and imagery stored and maintained in real time. Newly ingested images and videos are archived in S3 and accessed directly via S3 URLs.

The APIs used by client programs (UX or background) support a Contextual Data Model that offers various convenient features and returns filtered, transformed, and AI-enhanced slices of the database. These REST APIs are documented at [https://adology.ai/live/…](https://adology.ai/live/…).

The data stores (S3, Postgres, Mongo) that power the Adology Contextual Data Model are filled by Acquisition Engines, which are EC2 or dynamically launched Lambda functions that pull data from remote sources, including Facebook, SerpApi, and customer-specific databases.

Most of the data, such as brands and advertising assets, is shared by all customers, whereas brandspaces are private to each customer.

Different non-UI components run on either EC2 or Lambda functions and are connected through a series of SQS queues. The SQS queues and Lambdas support parallelization of workflows for maximum concurrency.

<!--BREAK-->
