# OD-growth-params-webapp

Completed web app that allows multiple files to be uploaded and analyzed to display an interactive graph.

The app allows for png downloads and tsv downloads.


temp_uploads will delete files over 24 hours. 


Run data into a csv converter. This app only accepts csv files. 

To run:

##Docker
Run these commands in the terminal:


docker build -t od-growth-app .


docker run -p 5000:5000 od-growth-app