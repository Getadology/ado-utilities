
//  counterdump.py
//  arco
//
//  Created by bill donner on 2/28/25.
//
"""

Below is an example of how you might design your MongoDB storage and a sample Python program to retrieve a daily report by date and render it as Markdown.

MongoDB Schema

In MongoDB you can use a single collection (e.g., daily_reports) where each document represents one day’s report. You can use the date as the document’s primary key (stored as the _id field). For example, using a Mongoose-like schema (if you were using Node.js) you could define it as follows:
"""

const mongoose = require('mongoose');

const DailyReportSchema = new mongoose.Schema({
  _id: { type: String, required: true }, // Date string, e.g., "1-March-24"
  settings: {
    step1_MaxLambda: { type: Number, required: true },
    step2_MaxLambda: { type: Number, required: true },
    step3_MaxLambda: { type: Number, required: true },
    step4_MaxLambda: { type: Number, required: true }
  },
  todays_counts: {
    new_brands_ingested: { type: Number, required: true },
    brands_updated: { type: Number, required: true },
    brands_fully_analyzed: { type: Number, required: true },
    images_processed: { type: Number, required: true },
    images_analyzed: { type: Number, required: true },
    videos_processed: { type: Number, required: true },
    videos_analyzed: { type: Number, required: true },
    other_assets_processed: { type: Number, required: true },
    step1_total_time: { type: Number, required: true },      // seconds
    step2_total_time: { type: Number, required: true },      // seconds
    step3_total_time: { type: Number, required: true },      // seconds
    step4_total_time: { type: Number, required: true },      // seconds
    total_ingestion_time: { type: Number, required: true }   // seconds
  },
  grand_totals: {
    total_brands: { type: Number, required: true },
    brands_fully_analyzed: { type: Number, required: true },
    total_images: { type: Number, required: true },
    images_analyzed: { type: Number, required: true },
    total_videos: { type: Number, required: true },
    videos_analyzed: { type: Number, required: true },
    total_other_assets: { type: Number, required: true },
    other_assets_analyzed: { type: Number, required: true }
  }
});

module.exports = mongoose.model('DailyReport', DailyReportSchema);
"""
Key points:
	•	Each document’s _id is the date (e.g., "1-March-24"), ensuring a single primary key per day.
	•	The document is structured into three parts: settings, todays_counts, and grand_totals.

Sample Python Program

Below is a sample Python program using pymongo. This script connects to a local MongoDB instance, retrieves the document for a given date, and prints a Markdown report.
"""

import sys
from pymongo import MongoClient

def generate_markdown_report(report):
    markdown = f"# Date: {report['_id']}\n\n"
    
    markdown += "## Settings\n"
    settings = report['settings']
    markdown += f"- **Step1 - Max Lambda:** {settings.get('step1_MaxLambda', settings.get('step1_Max Lambda', ''))}\n"
    markdown += f"- **Step2 - Max Lambda:** {settings.get('step2_MaxLambda', settings.get('step2_Max Lambda', ''))}\n"
    markdown += f"- **Step3 - Max Lambda:** {settings.get('step3_MaxLambda', settings.get('step3_Max Lambda', ''))}\n"
    markdown += f"- **Step4 - Max Lambda:** {settings.get('step4_MaxLambda', settings.get('step4_Max Lambda', ''))}\n\n"
    
    markdown += "## Today's Counts\n"
    counts = report['todays_counts']
    markdown += f"- **New Brands Ingested:** {counts['new_brands_ingested']}\n"
    markdown += f"- **Brands Updated:** {counts['brands_updated']}\n"
    markdown += f"- **Brands Fully Analyzed:** {counts['brands_fully_analyzed']}\n"
    markdown += f"- **Images Processed:** {counts['images_processed']}\n"
    markdown += f"- **Images Analyzed:** {counts['images_analyzed']}\n"
    markdown += f"- **Videos Processed:** {counts['videos_processed']}\n"
    markdown += f"- **Videos Analyzed:** {counts['videos_analyzed']}\n"
    markdown += f"- **Other Assets Processed:** {counts['other_assets_processed']}\n"
    markdown += f"- **Step1 Total Time:** {counts['step1_total_time']} seconds\n"
    markdown += f"- **Step2 Total Time:** {counts['step2_total_time']} seconds\n"
    markdown += f"- **Step3 Total Time:** {counts['step3_total_time']} seconds\n"
    markdown += f"- **Step4 Total Time:** {counts['step4_total_time']} seconds\n"
    markdown += f"- **Total Time to Complete the Ingestion:** {counts['total_ingestion_time']} seconds\n\n"
    
    markdown += "## Grand Totals in Adology Database\n"
    totals = report['grand_totals']
    markdown += f"- **Total Brands:** {totals['total_brands']}\n"
    markdown += f"- **Brands Fully Analyzed:** {totals['brands_fully_analyzed']}\n"
    markdown += f"- **Total Images:** {totals['total_images']}\n"
    markdown += f"- **Images Analyzed:** {totals['images_analyzed']}\n"
    markdown += f"- **Total Videos:** {totals['total_videos']}\n"
    markdown += f"- **Videos Analyzed:** {totals['videos_analyzed']}\n"
    markdown += f"- **Total Other Assets:** {totals['total_other_assets']}\n"
    markdown += f"- **Other Assets Analyzed:** {totals['other_assets_analyzed']}\n"
    
    return markdown

def main(date_str):
    # Connect to MongoDB (adjust connection string as needed)
    client = MongoClient("mongodb://localhost:27017/")
    db = client['adology_db']
    collection = db['daily_reports']
    
    # Retrieve the report by date (using the date as the _id)
    report = collection.find_one({"_id": date_str})
    
    if not report:
        print(f"No report found for date: {date_str}")
        return
    
    markdown_report = generate_markdown_report(report)
    print(markdown_report)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report.py <date>")
        sys.exit(1)
    date_input = sys.argv[1]
    main(date_input)
"""
How it works:
	•	The script connects to the adology_db database and the daily_reports collection.
	•	It looks up the document by its _id (the date string you pass as an argument).
	•	The generate_markdown_report function formats the data into Markdown.
	•	Finally, the Markdown report is printed to the console.

This setup provides a structured MongoDB schema with a primary key by date and a sample Python program to generate a daily report in Markdown format. Adjust connection strings and field names as needed for your environment.
"""
