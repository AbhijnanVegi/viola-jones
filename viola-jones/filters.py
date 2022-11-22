import numpy as np

from integral_image import integral_image

def regional_sum(iimg: np.ndarray, top_left: tuple, bottom_right: tuple) -> float:
    """
    Calculates the sum of a region given an integral image.
    @iimg: Integral image representation of image
    @top_left: tuple (x,y) representing top left corner of image
    @bottom_right: tuple (x,y) representing bottom right corner of image 
    """

    if top_left == bottom_right:
        return iimg[top_left]

    # calculate the sum of the region
    sum = iimg[bottom_right]

    if top_left[0] > 0:
        sum -= iimg[top_left[0]-1, bottom_right[1]]
    if top_left[1] > 0:
        sum -= iimg[bottom_right[0], top_left[1]-1]
    if top_left[0] > 0 and top_left[1] > 0:
        sum += iimg[top_left[0]-1, top_left[1]-1]

    return sum

def test_regional_sum():

    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    s = regional_sum(iimg, (1,1), (2,2))
    assert(s == 28)
    print("Regional sum test passed")

if __name__ == "__main__":
    test_regional_sum()