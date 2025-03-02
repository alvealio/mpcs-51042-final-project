import cv2
import numpy as np

def validate_image(image_path, min_width=100, min_height=100):
    """
    Validates that an image meets the following criteria: image can be loaded, meets minimum resolution,
    and has an appropriate number of channels.

    Args:
        image_path (str): The path to the image file.
        min_width (int, optional): Minimum allowed width (pixels).
        min_height (int, optional): Minimum allowed height (pixels).
    
    Returns:
        bool: True if image is valid.
    
    Raises:
        ValueError: If image cannot be read.
        ValueError: If resolution less than min_width x min_height.
        ValueError: If number of channels is unsupported (1, 3, 4).
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Image could not be loaded. Image is either an unsupported format or corrupted.')

    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        raise ValueError(f'Image resolution too small ({width}x{height}). Minimum required size: {min_width}x{min_height}')
    if len(image.shape) == 2:
        pass
    elif len(image.shape) == 3:
        channels = image.shape[2]
        # Supported channel counts: Grayscale (1), BGR (3), BGRA (4)
        if channels not in (1, 3, 4):
            raise ValueError(f'Number of channels is unsupported: {channels}')
    else:
        raise ValueError(f'Image shape is unsupported: {image.shape}')
    
    return True

def process_image(image_path, resize_width=500, padding=20, binary_threshold=128):
    """
    Converts an image to a processed binary image with the following operations:
    resizes while preserving aspect ratio, converts to grayscale, applies binary threshold,
    detects Canny edges, pads the image, dilates to close gapped edges, and flood fills.

    Args:
        image_path (str): The path to the image file.
        resize_width (int, optional): Target width for resizing.
        padding (int, optional): Number of pixels to pad the image.
        binary_threshold (int, optional): Threshold value for binary conversion.
    
    Returns:
        tuple: containing:
            processed_image (NumPy ndarray): Processed (filled) binary image.
            dimensions (int, int): The dimensions (width, height) of the processed image.

    Raises:
        ValueError: If image could not be loaded.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Image could not be loaded.')
    
    # Resize image while maintaining aspect ratio.
    (height, width) = image.shape[:2]
    scaled_height = int((width / width) * height)
    image_resized = cv2.resize(image, (resize_width, scaled_height))

    # Convert image to grayscale.
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold
    _, image_binary = cv2.threshold(image_gray, binary_threshold, 255, cv2.THRESH_BINARY)

    # Use Canny edge detection. 
    lower_bound = 100
    upper_bound = 200
    edges = cv2.Canny(image_binary, lower_bound, upper_bound)

    padded_edges = cv2.copyMakeBorder(
        edges,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=0
    )
    padded_height, padded_width = padded_edges.shape[:2]
    dimensions = (padded_width, padded_height)

    # Dilate image to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(padded_edges, kernel, iterations=2)

    floodfill_image = edges.copy()
    height, width = edges.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(floodfill_image, mask, (0, 0), 255)
    floodfill_image_inv = cv2.bitwise_not(floodfill_image)
    processed_image = edges | floodfill_image_inv

    return processed_image, dimensions

def extract_min_contours(image_path, resize_width=500, padding=20, threshold_init=200, threshold_step=-10, max_iter=20):
    """
    Extracts the minimal set of external contours from an image. Iteratively processes an image by 
    varying the binary threshold. In each iteration, the function checks the number of external 
    contours until a single contour is found. If greater than one contour is detected after max iterations,
    the minimal set of contours is returned. 

    Args:
        image_path (str): The path to the image file.
        resize_width (int, optional): Target width for resizing.
        padding (int, optional): Number of pixels to pad the image.
        threshold_init (int, optional): Initial threshold value for binary conversion.
        threshold_step (int, optional): Step size to increment the binary threshold.
        max_iter (int, optional): Maximum number of iterations to process image.
    
    Returns:
        tuple: containing:
            min_contours (list of ndarray): A list of contour(s) with shape (n, 1, 2).
            dimensions (int, int): The dimensions (width, height) of the processed image.
    
    Raises:
        ValueError: If no contours are found.
    """
    min_count  = np.inf
    min_contours = None
    for i in range(max_iter):
        threshold = threshold_init + i * threshold_step
        processed_image, dimensions = process_image(image_path, resize_width, padding, threshold)
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_count = len(contours)
        if contour_count < min_count and contour_count != 0:
            min_count = contour_count
            min_contours = contours
        if contour_count == 1:
            break
    if min_contours is None or len(min_contours) == 0:
        raise ValueError("Unable to extract any contours")
    
    # plot_contours(processed_image, min_contours)

    return min_contours, dimensions
    
def process_contours(contours):
    """
    Selects the largest contour by area in a list of contours, and returns this largest
    contour reshaped to two columns (Reshapes from (n, 1, 2) to (n, 2))

    Args:
        contours (list of ndarray): A list of contour(s) with shape (n, 1, 2).
    
    Returns:
        ndarray: A NumPy array of shape (n, 2) of the largest contour in contours.
    
    Raises:
        ValueError: If list of contours is empty.
    """
    if not contours:
        raise ValueError("No contours provided")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)

def plot_contours(processed_image, contours):
    """
    Development/debug function to visualize contours overlayed on 
    their processed image.

    Args:
        processed_image (NumPy ndarray): Processed (filled) binary image.
        contours (list of ndarray): A list of contour(s) with shape (n, 1, 2).
    """
    color_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        # Generate a random color (B, G, R)
        color = np.random.randint(0, 256, size=3).tolist()
        cv2.drawContours(color_image, [contour], -1, color, 2)
    cv2.imshow("Contours", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()