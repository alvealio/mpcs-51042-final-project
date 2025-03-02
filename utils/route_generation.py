import osmnx as ox
import networkx as nx
import pickle
import gzip
import gpxpy
import gpxpy.gpx


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

#def crop_chicago_graph(chicago_graph, bounds=(41.900500, 41.830983, -87.617354, -87.659620)):
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
        longitude = west + (horizontal_offset + x * scale_factor)
        # https://stackoverflow.com/questions/69392846/why-does-new-osmnx-nearest-nodes-function-return-different-results-than-old-fu
        nearest_node = ox.nearest_nodes(chicago_graph, longitude, latitude)
        route_nodes.append(nearest_node)
    # Remove duplicate nodes, but maintain order
    # https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    route_nodes = list(dict.fromkeys(route_nodes))
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
    # Make first node the last node to close the route
    if complete_route and complete_route[0] != complete_route[-1]:
        complete_route.append(complete_route[0])
    return complete_route

def generate_gpx(complete_route, chicago_graph, filename="route.gpx"):
    """ Generate gpx file from list of graph nodes"""
    # Create instance of GPX object
    gpx = gpxpy.gpx.GPX()
    # Create instance of GPXTrack (path)
    gpx_track = gpxpy.gpx.GPXTrack()
    # Append instance to tracks in GPX instance
    gpx.tracks.append(gpx_track)
    # Create instance of GPXTrackSegment
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    # Append to gpx_track 
    gpx_track.segments.append(gpx_segment)

    # Append each node from the chicago graph needed for our route to the segnebt
    for node in complete_route:
        point = chicago_graph.nodes[node]
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point['y'], point['x']))
    
    with open(filename, 'w') as f:
        f.write(gpx.to_xml())
    
    return filename
