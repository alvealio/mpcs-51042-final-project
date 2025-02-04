import cv2
import numpy as np

def validate_image(image_path):
    '''Examine image type, quality, and size to determine if viable input.
    '''
    pass

def process_image(image_path, size=(100, 100)):
    ''' Convert image to grayscale, resize, and return edges
    '''
    # Read image
    print(f'Reading {image_path} from disk')
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Image could not be loaded.')
    
    # Resize image (default 100x100px)
    print(f'Resizing image to {size[0]} x {size[1]} pixels')
    image_resized = cv2.resize(image, size)
    
    # Convert image to grayscale. Test with an already grayscale image. This might not work... may need to count channels
    print(f'Converting image to grayscale')
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Use binary threshold. Might want to make thresholds arguments.
    thresh = 128 
    maxval = 255 
    print(f'Applying binary threshold threshold value {thresh} and max value {maxval}')
    _, binary = cv2.threshold(image_gray, thresh, maxval, cv2.THRESH_BINARY)

    # Use Canny edge detection. Might also want Canny parameters to be arguments
    lower_bound = 100 # Gradients below this threshold are non-edges
    upper_bound = 200 # Gradients above this threshold are strong edges
    print(f'Detecting edges with lower and upper bounds of {lower_bound} and {upper_bound}, respectively')
    edges = cv2.Canny(binary, lower_bound, upper_bound)

    return edges


process_image('star.png')