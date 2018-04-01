class BoundingBox:
    def __init__(self, left, bottom, right, top, image_width, image_height, label): #swap tob and bottom
        self.left = max(0, left)
        self.top = min(top, image_height)
        self.right = min(right, image_width)
        self.bottom = max(0, bottom)
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.image_width = image_width
        self.image_height = image_height
        self.label = label

    def __repr__(self):
        return '(x1: {}, y1: {}, x2: {}, y2: {} ({}))'.format(self.left, self.top, self.right, self.bottom, self.label)

    def flip(self):
        left = self.image_width - self.right
        top = self.image_height - self.bottom
        right = self.image_width - self.left
        bottom = self.image_height - self.top
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        return self

    def resize(self, width, height):
        width_ratio = width / self.image_width
        height_ratio = height / self.image_height
        self.left = int(self.left * width_ratio)
        self.top = int(self.top * height_ratio)
        self.right = int(self.right * width_ratio)
        self.bottom = int(self.bottom * height_ratio)
        return self
