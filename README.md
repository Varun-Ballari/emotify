# emotify

Emotify lets you identify people in real time and play songs based on their detected facial expressions!

Disclaimer: this application is not representative of true emotion and should not be used for any purpose other than fun.

#### Instructions to add new person in classifier:
1. Open the ExpressMusic/cv/take_images.py file
2. Change person variable to the name of the person (No spaces, no commas etc)
3. Assign a new unique person_number
4. In the 'training-data' folder, create a new folder whose name is the unique person_number.
5. Run the following commands: <br>
      (i) "cd ExpressMusic/cv" <br>
      (ii) "python3 take_images.py"
6. In cameraCV.py, add an entry to the 'people' dictionary with key=unique_number and value=Name of the new person

Created by Team Compiling... for a Hackathon.
