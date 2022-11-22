import numpy as np

from integral_image import integral_image

def regional_sum(iimg: np.ndarray, top_left: tuple, bottom_right: tuple) -> float:
    """
    Calculates the sum of a region given an integral image.
    @iimg: Integral image representation of image
    @top_left: tuple (x,y) representing top left corner of image
    @bottom_right: tuple (x,y) representing bottom right corner of image 
    """
    # calculate the sum of the region
    sum = iimg[bottom_right]

    if top_left[0] > 0:
        sum -= iimg[top_left[0]-1, bottom_right[1]]
    if top_left[1] > 0:
        sum -= iimg[bottom_right[0], top_left[1]-1]
    if top_left[0] > 0 and top_left[1] > 0:
        sum += iimg[top_left[0]-1, top_left[1]-1]

    return sum

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.tl = (x, y)
        self.tr = (x, y + w)
        self.bl = (x + h, y)
        self.br = (x + h, y + w)
    

class HaarFilter:
    """
    Define a filter class that can be used to calculate the sum of a region
    @width: Width of the filter
    @height: Height of the filter
    @white_rect: List of Rects representing white rectangles
    @black_rect: List of Rects representing black rectangles
    """
    def __init__(self, white_rect, black_rect):
        self.white_rect = white_rect
        self.black_rect = black_rect

    def apply(self, iimg, x, y):
        white_sum:float = 0
        black_sum:float = 0

        for rect in self.white_rect:

            top_left = (x + rect.x ,y + rect.y)
            bottom_right = (x + rect.br[0],y + rect.br[1])

            white_sum += regional_sum(iimg, top_left, bottom_right)

        for rect in self.black_rect:

            top_left = (x + rect.x ,y + rect.y)
            bottom_right = (x + rect.br[0],y + rect.br[1])

            black_sum += regional_sum(iimg, top_left, bottom_right)

        return white_sum - black_sum


class TwoColumnFilter(HaarFilter):
    def __init__(self, width, height):
        if (width %2 != 0):
            raise ValueError("Width must be even")
        super().__init__([Rect(0,0,width//2-1,height - 1)], [Rect(0,width//2,width//2 - 1,height - 1)])

class TwoRowFilter(HaarFilter):
    def __init__(self, width, height):
        if (height %2 != 0):
            raise ValueError("Height must be even")
        super().__init__([Rect(0,0,width-1,height//2 - 1)], [Rect(height//2,0,width-1,height//2 - 1)])

class ThreeColumnFilter(HaarFilter):
    def __init__(self, width, height):
        if width%3 != 0:
            raise ValueError("Width must be divisible by 3")

        super().__init__([Rect(0,0,width//3 - 1,height - 1), Rect(0,2*width//3,width//3 - 1,height - 1)], [Rect(0,width//3,width//3 - 1,height - 1)])

class QuadFilter(HaarFilter):
    def __init__(self, width, height):
        if width != height:
            raise ValueError("Width and height must be equal")
        if width%2 != 0:
            raise ValueError("Width and height must be even")
        super().__init__([Rect(0,0,width//2 - 1,height//2 - 1), Rect(height//2,width//2,width//2 - 1,height//2 - 1)],
                         [Rect(0,height//2,width//2 - 1,height//2 - 1), Rect(width//2,0,width//2 - 1,height//2 - 1)])


def test_regional_sum():

    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    s = regional_sum(iimg, (1,1), (2,2))
    assert(s == 28)
    print("Regional sum test passed")

def test_two_column():
    # img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = TwoColumnFilter(2, 2)
    score = filter.apply(iimg, 0, 0)
    assert(score == -2)
    print("Two column filter test passed")

def test_two_row():
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = TwoRowFilter(2, 2)
    score = filter.apply(iimg, 0, 0)

    assert(score == -6)
    print("Two row filter test passed")

def test_three_column():
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = ThreeColumnFilter(3, 3)
    score = filter.apply(iimg, 0, 0)

    assert(score == 15)
    print("Three column filter test passed")

def test_quad_filter():
    iimg = np.array([[1, 3, 6], [5, 12, 21], [12, 27, 45]])

    filter = QuadFilter(2, 2)
    score = filter.apply(iimg, 0, 0)

    assert(score == 0)
    print("Quad filter test passed")


if __name__ == "__main__":
    test_regional_sum()
    test_two_column()
    test_two_row()
    test_three_column()
    test_quad_filter()
            