import os
import osmnx as ox
import networkx as nx
import pickle
import gzip
import gpxpy
import gpxpy.gpx

def get_map_graph(place="Chicago, Illinois, USA", filename='chicago_walk_graph.pkl.gz'):
    """
    Retrieves a map graph of Chicago (default) with OSMnx. Stores this graph in a compressed pickle file.

    Args:
        place (str, optional): Location for map graph retrieval.
        filename (str, optional): Filename of graph stored.
    
    Returns:
        networkx.MultiDiGraph: A directed graph of walkable streets/paths in specified location.
    """
    if os.path.exists(filename):
        with gzip.open(filename, 'rb') as f:
            graph = pickle.load(f)
    else:
        custom_filter = (
        '["highway"~"footway|pedestrian|path|living_street|residential|secondary|primary"]'
        '["highway"!~"service|alley"]'
        )
        graph = ox.graph_from_place(place, network_type='walk', simplify=True, custom_filter=custom_filter)
        with gzip.open(filename, "wb") as f:
            pickle.dump(graph, f)
    
    return graph

def crop_map_graph(graph, bounds=(41.900500, 41.830983, -87.617354, -87.659620)):
    """
    Crops map graph to a specified bounding box.

    Args:
        graph (networkx.MultiDiGraph): A directed graph of walkable streets/paths in specified location.
        bounds (tuple, optional): A tuple (north, south, east, west) defining the crop boundaries.

    Returns:
        networkx.MultiDiGraph: Cropped map graph
    """
    north, south, east, west = bounds
    bounding_box = (west, south, east, north)

    return ox.truncate.truncate_graph_bbox(graph, bounding_box)

def contour_to_route(contour, graph, dimensions, bounds=(41.900500, 41.830983, -87.617354, -87.659620)):
    """
    Maps contour points to map graph to generate a running route. Normalizes contour points to geographic 
    coordinates then locates the nearest nodes in the map graph to construct the route.

    Args:
        contour (list of tuples): A list of (x, y) coordinates corresponding to contours of an image.
        graph (networkx.MultiDiGraph): A directed graph of walkable streets/paths in specified location.
        dimensions (int, int): The dimensions (width, height) of the processed image.
        bounds (tuple of floats): A tuple (north, south, east, west) defining the contour mapping area.

    Returns:
        list: A list of map graph node IDs
    """
    width, height = dimensions
    north, south, east, west = bounds
    bound_width = east - west
    bound_height = north - south

    # Compute a scale factor to preserve aspect ratio
    scale_factor = min(bound_width / width, bound_height / height)
    horizontal_offset = (bound_width - width * scale_factor) / 2
    vertical_offset = (bound_height - height * scale_factor) / 2
    
    route_nodes = []
    for (x, y) in contour:
        latitude = north - (vertical_offset + y * scale_factor)
        longitude = west + (horizontal_offset + x * scale_factor)
        nearest_node = ox.nearest_nodes(graph, longitude, latitude)
        route_nodes.append(nearest_node)
    # Remove duplicate nodes while preserving order
    route_nodes = list(dict.fromkeys(route_nodes))

    # Connect each node with a series of closest nodes
    complete_route = []
    for i in range(len(route_nodes) - 1):
        try:
            path = nx.shortest_path(graph, route_nodes[i], route_nodes[i + 1], weight="length")
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

def generate_gpx(route, graph, gpx_folder, filename="route.gpx"):
    """
    Generate a GPX file from list of graph nodes in a map graph.
    
    Args:
        route (list): A list of map graph node IDs
        graph (networkx.MultiDiGraph): A directed graph of walkable streets/paths in specified location.
        gpx_folder (str): Directory to save GPX file.
        filename (str, optional): Filename of the GPX file.

    Returns:
        str: Filename of the GPX file.
    """
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for node in route:
        point = graph.nodes[node]
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(point['y'], point['x']))
    
    file_path = os.path.join(gpx_folder, filename)
    with open(file_path, 'w') as f:
        f.write(gpx.to_xml())
    
    return filename
