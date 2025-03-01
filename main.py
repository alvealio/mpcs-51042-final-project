import cv2
import numpy as np
import osmnx as ox
import networkx as nx
import pickle
import gzip
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def validate_image(image_path):
    '''Examine image type, quality, and size to determine if viable input.
    '''
    # We should probably read image here then pass to process_image

    # Check image type

    # Check image size

    # Check image channels?
    pass

def process_image(image_path, width=500, padding=20):
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

    # Pad the image to ensure edges are on the border

    # Use Canny edge detection. Might also want Canny parameters to be arguments
    lower_bound = 100 # Gradients below this threshold are non-edges
    upper_bound = 200 # Gradients above this threshold are strong edges
    edges = cv2.Canny(binary, lower_bound, upper_bound)

    padded_edges = cv2.copyMakeBorder(
        edges,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=0
    )
    padded_height, padded_width = padded_edges.shape[:2]

    return padded_edges, (padded_width, padded_height)

def extract_contours(edges, area_threshold=1):

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    im_floodfill = edges.copy()
    h, w = edges.shape[:2]

    # Create a mask that is 2 pixels larger than the image
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from a corner (assuming the corner is background)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    
    # Invert floodfilled image: now the filled area becomes black and the original
    # white shape is preserved.
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the original binary image with the inverted floodfilled image.
    # This results in an image where the interior of the shape is filled.
    filled = edges | im_floodfill_inv
    # Area threshold should be a function of image size
    # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    # Retrieve all contours (RETR_LIST) with simple chain approximation to minimize points
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours detected")
    print(cv2.contourArea(contours[0]))
    print(f'There are {len(contours)} contours')
    for i, cnt in enumerate(contours):
        print(f'Contour {i} is length {len(cnt)}: {cnt}')

    # Filter out small contours and reshape
    contours_filtered = [x.reshape(-1, 2) for x in contours] # if cv2.contourArea(x) > area_threshold]
    if not contours_filtered:
        raise ValueError("No contours of adequate size")
    print(f'After filtering, there are {len(contours_filtered)} contours')
    
    color_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for cnt in contours_filtered:
        # Generate a random color (B, G, R)
        color = np.random.randint(0, 256, size=3).tolist()
        cv2.drawContours(color_img, [cnt], -1, color, 2)

    cv2.imshow("Contours with Different Colors", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return np.vstack(contours_filtered)

def get_chicago_graph():
    # Note this takes a long time (Chicago is large...). Let's save the result with pickle and only call this function if that pickle file does not exist

    place = "Chicago, Illinois, USA"

    custom_filter = (
    '["highway"~"footway|pedestrian|path|living_street|residential|secondary|primary"]'
    '["highway"!~"service|alley"]'
    )
    # Set network_type to 'walk' so this doesn't return non-runnable paths
    graph = ox.graph_from_place(place, network_type='walk', simplify=True, custom_filter=custom_filter)

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

def image_to_route(contours, chicago_graph, dimensions, bounds=(41.900500, 41.830983, -87.617354, -87.659620)):
    """
    Map contours to a running route
    """
    # I could improve the mapping by running this multiple times from different starting points and picking the graph with lowest error
    # Store sum of errors (distances between node and points)
    
    width, height = dimensions
    north, south, east, west = bounds
    bound_width = east - west
    bound_height = north - south

    # Make a scale factor so image isn't stretched 
    scale_factor = min(bound_width / width, bound_height / height)

    # Determine offsets
    horizontal_offset = (bound_width - width * scale_factor) / 2
    vertical_offset = (bound_height - height * scale_factor) / 2
    
    # Treat each ndarray element as a coordinate normalized to the height and width of the image
    route_nodes = []
    for (x, y) in contours:
        # Calculate latitude and longitude for each point
        latitude = north - (vertical_offset + y * scale_factor)
        longitude = east - (horizontal_offset + x * scale_factor)
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
    # get_chicago_graph()
    edges, dimensions = process_image('images/heart.jpg')
    contours = extract_contours(edges)
    chicago_graph = plot_chicago_graph()
    cropped_graph = crop_chicago_graph(chicago_graph)
    full_route = image_to_route(contours, cropped_graph, dimensions)
    generate_gpx(full_route, cropped_graph)

    animate_contour(contours, interval=50)
