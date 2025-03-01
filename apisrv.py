#
//  apisrv.py
//  arco
//
//  Created by bill donner on 2/28/25.
//


# Below is an updated Flask server that implements the /api/todaysreport endpoint exactly as expected by the JavaScript client. It connects to MongoDB, retrieves the report document for a given date (defaulting to today’s date), and returns a JSON response with the structure required by the client (including keys: date, settings, todays_counts, and grand_totals).

from flask import Flask, jsonify, request
from pymongo import MongoClient
import datetime

app = Flask(__name__)

# Connect to MongoDB (adjust the URI as needed)
client = MongoClient("mongodb://localhost:27017/")
db = client['adology_db']
collection = db['daily_reports']

def format_date(dt):
    """
    Format a datetime object as a string matching the _id format in the database.
    For example, March 1, 2024 becomes "1-March-24".
    """
    return f"{dt.day}-{dt.strftime('%B')}-{dt.strftime('%y')}"

@app.route('/api/todaysreport', methods=['GET'])
def todays_report():
    # Use the provided 'date' query parameter, or default to today's date.
    date_str = request.args.get("date")
    if not date_str:
        date_str = format_date(datetime.datetime.today())
    
    # Retrieve the report document using the date as the primary key (_id)
    report = collection.find_one({"_id": date_str})
    
    if not report:
        return jsonify({"error": f"No report found for date: {date_str}"}), 404
    
    # Rename _id to date to match the expected JSON structure
    report["date"] = report.pop("_id")
    
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True)

""" How It Works
	•	Endpoint:
The /api/todaysreport endpoint returns the report document in JSON format. The endpoint accepts an optional date query parameter (e.g., /api/todaysreport?date=1-March-24) or defaults to today’s date if not provided.
	•	Database Query:
The code queries the daily_reports collection using the date string as the primary key (_id). The document is then modified by renaming _id to date.
	•	JSON Structure:
The returned JSON contains keys for date, settings, todays_counts, and grand_totals—exactly as expected by the JavaScript client.

This Flask server, in conjunction with the previously generated JavaScript client, will allow your HTML page to fetch and render the report on a minute-by-minute basis.
