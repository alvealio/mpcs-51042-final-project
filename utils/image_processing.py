import cv2
import numpy as np

def validate_image(image_path, min_width=100, min_height=100):
    '''Examine image type, quality, and size to determine if viable input.
    '''
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Image could not be loaded. Image is either an unsupported format or corrupted.')

    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        raise ValueError(f'Image resolution too small ({width}x{height}). Minimum required size: {min_width}x{min_height}')
    if len(image.shape) == 2:
        # Grayscale images 
        pass
    elif len(image.shape) == 3:
        channels = image.shape[2]
        # Grayscale (1), BGR (3), BGRA (4)
        if channels not in (1, 3, 4):
            raise ValueError(f'Number of channels is unsupported: {channels}')
    else:
        raise ValueError(f'Image shape is unsupported: {image.shape}')
    
    return True


def process_image(image_path, width=500, padding=20, binary_threshold=128):
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
    max_val = 255 
    _, binary = cv2.threshold(image_gray, binary_threshold, max_val, cv2.THRESH_BINARY)

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

    # Dilate image to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(padded_edges, kernel, iterations=2)
    
    # Flood fill
    im_floodfill = edges.copy()
    h, w = edges.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    filled = edges | im_floodfill_inv

    return filled, (padded_width, padded_height)

def extract_min_contours(image_path, width=500, padding=20, threshold_init=200, threshold_step=-10, max_iter=20):
    min_count  = np.inf
    min_contours = None
    for i in range(max_iter):
        threshold = threshold_init + i * threshold_step
        processed_image, dims = process_image(image_path, width, padding, threshold)
        # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)

        if contour_count < min_count and contour_count != 0:
            min_count = contour_count
            min_contours = contours
        print(f'Found {len(contours)} contours')

        if contour_count == 1:
            break
    if min_contours is None or len(min_contours) == 0:
        raise ValueError("Unable to extract any contours")
    
    plot_contours(processed_image, min_contours)

    return min_contours, dims
    
def process_contours(contours):
    if not contours:
        raise ValueError("No contours provided")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)

def plot_contours(edges, contours):
    color_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        # Generate a random color (B, G, R)
        color = np.random.randint(0, 256, size=3).tolist()
        cv2.drawContours(color_img, [cnt], -1, color, 2)
    cv2.imshow("Contours with Different Colors", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()