import numpy as np

from integral_image import integral_image

def regional_sum(iimg: np.ndarray, top_left: tuple, bottom_right: tuple) -> float:
    """
    Calculates the sum of a region given an integral image.
    Doesn't include row and column of top_left.
    @iimg: Integral image representation of image
    @top_left: tuple (x,y) representing top left corner of image
    @bottom_right: tuple (x,y) representing bottom right corner of image 
    """
    top_left = (top_left[0]-1, top_left[1]-1)

    if top_left == bottom_right:
        return iimg[top_left]

    # calculate other endpoints of the region
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    return iimg[top_left] + iimg[bottom_right] - (iimg[top_right] + iimg[bottom_left])

def test_regional_sum():

    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    s = regional_sum(iimg, (0,0), (2,2))
    print(s)

if __name__ == "__main__":
    test_regional_sum()