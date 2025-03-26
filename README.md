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


## TODOs

- Getting raspberry, webcam and maybe scale?

- Finding opencv libs
