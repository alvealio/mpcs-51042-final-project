import cv2
import numpy as np
import osmnx as ox
import networkx as nx
import pickle
import gzip
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt

def validate_image(image_path):
    '''Examine image type, quality, and size to determine if viable input.
    '''
    # We should probably read image here then pass to process_image

    # Check image type

    # Check image size

    # Check image channels?
    pass

def process_image(image_path, width=500):
    ''' Convert image to grayscale, resize, and return edges
        Depending on certain image qualities examined in validate_image, we may want to pass different width args depending on image size and details
    '''
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Image could not be loaded.')
    
    # Resize image while maintaining aspect ratio. Let width constrain the dimensions and calculate the height
    (h, w) = image.shape[:2]
    scaled_height = int((width / w) * h)
    image_resized = cv2.resize(image, (width, scaled_height))
    
    # Convert image to grayscale. Test with an already grayscale image. This might not work... may need to count channels
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Use binary threshold. Might want to make thresholds arguments.
    thresh = 128
    max_val = 255 
    _, binary = cv2.threshold(image_gray, thresh, max_val, cv2.THRESH_BINARY)

    # Use Canny edge detection. Might also want Canny parameters to be arguments
    lower_bound = 100 # Gradients below this threshold are non-edges
    upper_bound = 200 # Gradients above this threshold are strong edges
    edges = cv2.Canny(binary, lower_bound, upper_bound)

    # Preview edges
    cv2.imshow('Canny Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edges

def sample_contour(edges, sample_rate=5, area_threshold=2):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours detected")

    # Filter out small contours
    contours_filtered = [x for x in contours if cv2.contourArea(x) > area_threshold]
    if not contours_filtered:
        raise ValueError("No contours of adequate size")
    
    # Merge points from each contour
    combined_points = []
    for contour in contours_filtered:
        points = contour.reshape(-1,2)
        combined_points.append(points)
    
    # Combine points vertically into a single array
    combined_points = np.vstack(combined_points)

    # Sample points on the contour
    sampled_points = combined_points[::sample_rate]
    print(f'Number of sample points: {len(sampled_points)}')
    print(sampled_points)
    return sampled_points


def get_chicago_graph():
    # Note this takes a long time (Chicago is large...). Let's save the result with pickle and only call this function if that pickle file does not exist

    place = "Chicago, Illinois, USA"
    # Set network_type to 'walk' so this doesn't return non-runnable paths
    graph = ox.graph_from_place(place, network_type='walk', simplify=True)

    # Compress and pickle the graph
    with gzip.open("chicago_walk_graph.pkl.gz", "wb") as f:
        pickle.dump(graph, f)

def plot_chicago_graph():
    CHICAGO_GRAPH_NAME = 'chicago_walk_graph.pkl.gz'
    with gzip.open(CHICAGO_GRAPH_NAME, 'rb') as f:
        graph = pickle.load(f)

    return graph
    # ox.plot_graph(ox.project_graph(graph))

def crop_chicago_graph(chicago_graph, bounds=(41.900500, 41.830983, -87.617354, -87.659620)):
    # Bounds is a tuple of north, south, east, west
    north, south, east, west = bounds
    bounding_box = (west, south, east, north)
    return ox.truncate.truncate_graph_bbox(chicago_graph, bounding_box)

def image_to_route(edges, chicago_graph, bounds=(41.900500, 41.830983, -87.617354, -87.659620)):
    """
    Map array edges to a running route
    """
    # Get coordinate bounds from chicago_graph. Might need this to be dynamic later.
    # Need to convert graph to GeoDataFrames
    sampled_points = sample_contour(edges)

    width, height = edges.shape
    north, south, east, west = bounds

    # Treat each ndarray element as a coordinate normalized to the height and width of the image
    route_nodes = []
    for (x, y) in sampled_points:
        # Calculate latitude and longitude for each point
        latitude = north - (y / height) * (north - south)
        longitude = east - (x / width) * (east - west)
        # https://stackoverflow.com/questions/69392846/why-does-new-osmnx-nearest-nodes-function-return-different-results-than-old-fu
        nearest_node = ox.nearest_nodes(chicago_graph, longitude, latitude)
        route_nodes.append(nearest_node)
    # Remove duplicate nodes, but maintain order
    # https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    route_nodes = list(dict.fromkeys(route_nodes))
    print(route_nodes)
    # Connect each node with a series of closest nodes
    complete_route = []
    for i in range(len(route_nodes) - 1):
        try:
            path = nx.shortest_path(chicago_graph, route_nodes[i], route_nodes[i + 1], weight="length")
            # Prevent node duplication
            if i > 0:
                path = path[1:]
            complete_route.extend(path)
        except nx.NetworkXNoPath:
            continue
    print(complete_route)
    return complete_route

def generate_gpx(complete_route, chicago_graph, filename="route.gpx"):
    """ Generate gpx file from list of graph nodes"""
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for node in complete_route:
        point = chicago_graph.nodes[node]
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point['y'], point['x']))
    
    with open(filename, 'w') as f:
        f.write(gpx.to_xml())
    
    return filename

if __name__ == '__main__': 
    edges = process_image('images/heart.jpg')
    # chicago_graph = plot_chicago_graph()
    # cropped_graph = crop_chicago_graph(chicago_graph)
    # full_route = image_to_route(edges, cropped_graph)
    # generate_gpx(full_route, cropped_graph)
