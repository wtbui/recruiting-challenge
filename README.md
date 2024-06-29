# IdentifAI Recruiting Challenge - William Bui

Objective: Develop a FastAPI application that allows users to upload an image and generates a "facial profile" that describes key characteristics of the face which define its reality.
Timeframe: You have until June 29th! Just let us know when you start so we can keep track. The project in total shouldn't take longer than 3 days, however we understand some may have busy schedules!

Tasks:
1. Setup FastAPI Server: Develop a basic FastAPI server to handle image uploads.
2. Facial Analysis: Implement a method (or use a library) to analyze facial features from the uploaded image.
3. API Endpoint: Create an endpoint that receives an image, processes it to extract facial features, and saves a "profile" of these features.
4. Create a use case showing how this "facial profile" could be used to identify a separate image as real.

## Getting Started
### 1. Clone this repo. <br/>
```
git clone https://github.com/wtbui/recruiting-challenge.git
```
### 2. Install dependencies. <br/>
```
pip install -r requirements.txt
```
### 3. Run FastAPI server. 
Note: MUST run from project directory with app. before main:app<br/>
```
uvicorn app.main:app --reload
```
<br/>
 <br/>
### 4. Try out the api! 
Note: Data is not persistent among calls, will need database integration <br/>
```
http://localhost:8000/docs
```



