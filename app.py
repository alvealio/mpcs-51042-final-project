from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
import os
import uuid
from utils.image_processing import validate_image, extract_min_contours, process_contours
from utils.route_generation import get_map_graph, crop_map_graph, contour_to_route, generate_gpx

app = Flask(__name__)
app.secret_key = "1234" 

UPLOAD_FOLDER = "uploads"
GPX_FOLDER = "gpx"

# Create upload and download folders if they do not already exist
for folder in (UPLOAD_FOLDER, GPX_FOLDER):
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GPX_FOLDER"] = GPX_FOLDER

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for selecting map boundaries
@app.route("/select_bounds", methods=["GET", "POST"])
def select_bounds():
    if request.method == "POST":
        try:
            north = float(request.form.get("north"))
            south = float(request.form.get("south"))
            east = float(request.form.get("east"))
            west = float(request.form.get("west"))
            session["bounds"] = (north, south, east, west)
            flash("Successfully defined map boundaries!")
            return redirect(url_for("process_image_route"))
        except:
            flash("Invalid bounds selected. Please try again.")
            return redirect(url_for("select_bounds"))
    return render_template("select_bounds.html")

# Route to process an image and select map bounds
@app.route("/process_image", methods=["GET", "POST"])
def process_image_route():
    # Check that user has defined map boundaries in the session
    if "bounds" not in session:
        flash("Please select map boundary first.")
        return redirect(url_for("select_bounds"))

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No image file selected.")
            return redirect(request.url)
        # Generate a random uuid for each image uploaded
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        
        # Run image processing and route generation functions
        try:
            validate_image(file_path)
            contours, dims = extract_min_contours(file_path)
            contour = process_contours(contours)
            chicago_graph = get_map_graph()
            cropped_graph = crop_map_graph(chicago_graph, session["bounds"])
            route = contour_to_route(contour, cropped_graph, dims, session["bounds"])
            gpx_filename = generate_gpx(route, cropped_graph, app.config["GPX_FOLDER"])
            return redirect(url_for("result", filename=gpx_filename))
        except Exception as e:
            flash(str(e))
            return redirect(request.url)
    return render_template("process_image.html")

# Route to download the generated GPX file
@app.route("/downloads/<filename>")
def download_gpx(filename):
    return send_from_directory(app.config["GPX_FOLDER"], filename, as_attachment=True)

# Route to result page containing GPX file download link
@app.route("/result")
def result():
    filename = request.args.get("filename")
    return render_template("result.html", filename=filename)

if __name__ == "__main__":
    app.run(debug=True)