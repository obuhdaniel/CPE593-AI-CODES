import numpy as np
kernel = np.array([
    [2, 0, -1],
    [2, 0, -1],
    [2, 0, -1]
])

def convolve_3x3(image_patch):
    """
    Perform convolution on a 3x3 image patch using a fixed 3x3 kernel.
    
    Args:
        image_patch (list[list[int]] or np.ndarray): 3x3 image patch

    Returns:
        int: Result of the convolution
    """
    image_patch = np.array(image_patch)
    if image_patch.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")
    
    result = np.sum(image_patch * kernel)
    return result

input_patch = [
    [1, 5, 8],
    [6, 1, 8],
    [2, 6, 9]
]

print("Convolution Result:", convolve_3x3(input_patch))
