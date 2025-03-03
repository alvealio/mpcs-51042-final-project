from utils.image_processing import *
from utils.route_generation import *

if __name__ == '__main__': 
    # get_chicago_graph()
    image_path = 'images/pikachu.png'
    if validate_image(image_path):
        contours, dims = extract_min_contours(image_path)
        contour = process_contours(contours)
        chicago_graph = get_map_graph()
        cropped_graph = crop_chicago_graph(chicago_graph)
        full_route = image_to_route(contour, cropped_graph, dims)
        generate_gpx(full_route, cropped_graph)
