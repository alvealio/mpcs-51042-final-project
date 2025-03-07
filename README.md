# mpcs-51042-final-project
Generates custom Strava art by transforming user-uploaded images into runnable routes on Chicago streets, leveraging image processing, street graph mapping, and GPX file creation.

**Proposal**
* I would like to build an application that allows Chicago runners to import custom running routes to Strava that depict the edges of a desired image i.e., Strava art. It will contain methods to:
    - Validate and process the user’s image
    - Return the prominent image edges with OpenCV
    - Return graph of Chicago for viable walking/running routes with OSMnx
    - Map the image edges to the Chicago street graph
    - Create a GPX file so we can import to Strava
    - Import GPX file to user’s Strava account using Strava’s API (Note: not supported by Strava API)
    - Optional: build a front-end for user image upload and visualization
    - For context, here are some manually created pieces of Strava Art: 
    <https://stories.strava.com/articles/love-strava-art-how-to-create-your-masterpiece>

**Execution Plan**
- Week 4: Implement methods that process user images and return image edges with OpenCV. Experiment with OSMnx Chicago street graph output to confirm compatibility with edge output. Read Flask documentation to understand how to implement back-end.
- Week 5: If OSMnx is not viable, confirm a new map graph data source. Implement methods to map image edges to Chicago street graph. Read Strava API documentation and test API calls. 
- Week 6: Implement methods to create a GPX file from mapped edges and to import GPX to a Strava account (Note: not supported by Strava API). 
- Week 7: Test methods with a large sample of images and optimize image processing parameters. Complete back-end implementation and testing (back-end will be hosted on my local machine)
- Week 8: Implement a front-end (locally hosted page for upload and visualization) if time allows. Otherwise, finalize testing and debugging.

# Installation
1. Clone this repository.
2. Install required Python packages in requirements.txt (virtual environment recommended)
3. Download the [Chicago Map Graph](https://drive.google.com/file/d/1evbrh8LIOWXiz2ZNmvMtsWmgumY0_CD7/view?usp=sharing) and place in root directory of the project. Note: if the graph file is not present, the app will call OSMnx to generate it which may take some time.
4. Run app.py (App should now be running locally at http://127.0.0.1:5000)

