# digital-signal-processing

## System Design

The basic idea is that a person places the parcel in a device, closes the door, presses a button to confirm everything, 
and then the system flow begins. 
Inside the device, the scale transfers the weight of the box to the Raspberry Pi via an analogue signal.
If we can't get our hands on such a scale, we will try to get a scale that displays the weight, and the camera will 
also track the weight.
A camera is attached to the ceiling inside the device and scans for the box ID and the tracking ID number. 
This scanning is achieved through OpenCV and Python. Afterwards, the Pi sends the scanned data to a server via an API. 
The server handles the incoming requests and saves the data in an Excel sheet. An extension would be to host a web 
server and a database to display the data even better.
![alt text](SystemDesign.png "Our System Design")

### Needed Hardware
- WebCam with at least 720p

- Scale (LoadCell directly connected or with Display for WebCam Scan)

- RaspberryPI default components (Wifi for server connection)

- PC for Server

### Software

#### Client
- Raspbian (OS)
  - OpenCV + Python

#### Server
- Linux
  - PostgreSQL
  - NodeJS + React for simple WebInterface


# How to run our code
We created a requirements file with all the pip requirements to run our Code locally!
Create a new Python virtual environment in the project directory by running this command(https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/):

For Unix/macOS <br />

`python3 -m venv .venv`<br />

For Windows<br />

`py -m venv .venv`

Then activate it by using this command <br />

Unix/macOS<br />

`source .venv/bin/activate`<br />

Windows<br />

`.venv\Scripts\activate`

Afterwards, install our requirements:

Unix/macOS/Windows

`pip install -r requirements.txt`

Now the project should be runnable by executing the main.py Python script:

`python main.py`


Additionally, if you encounter errors, one source could be the OpenCV libraries! Although not needed, you could install OpenCV on the machine too. (Please see Iclass)

GUI
Since our Code includes the GUI in the Code, if you want to run it only for the analysis on a dataset folder next to our main.py, you could set the DEBUG flag inside the main.py to True. This will then run the analysis on the pictures found in a "dataset" directory ("csv_path = analyze_image("dataset")").



## TODOs

- Getting raspberry, webcam and maybe scale?

- Finding opencv libs

- Change README to include model 
