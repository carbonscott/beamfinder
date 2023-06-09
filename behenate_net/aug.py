import numpy as np
import random

from skimage.transform import rotate, resize


class PadBottomRight:
    def __init__(self, size_y, size_x):
        self.size_y = size_y
        self.size_x = size_x


    def __call__(self, img, center):
        size_y = self.size_y
        size_x = self.size_x

        size_img_y, size_img_x = img.shape

        dy_padded = max(size_y - size_img_y, 0)
        dx_padded = max(size_x - size_img_x, 0)
        pad_width = (
            (0, dy_padded),
            (0, dx_padded),
        )

        img_padded = np.pad(img, pad_width = pad_width, mode = 'constant', constant_values = (0, 0))

        return img_padded, center




class Pad:
    def __init__(self, size_y, size_x):
        self.size_y = size_y
        self.size_x = size_x


    def __call__(self, img, center):
        size_y = self.size_y
        size_x = self.size_x

        size_img_y, size_img_x = img.shape

        dy_padded = max(size_y - size_img_y, 0)
        dx_padded = max(size_x - size_img_x, 0)
        pad_width = (
            (dy_padded // 2, dy_padded - dy_padded // 2),
            (dx_padded // 2, dx_padded - dx_padded // 2),
        )

        img_padded = np.pad(img, pad_width = pad_width, mode = 'constant', constant_values = (0, 0))

        cy, cx = center
        cy_padded = cy + dy_padded // 2
        cx_padded = cx + dx_padded // 2
        center_padded = (cy_padded, cx_padded)

        return img_padded, center_padded




class Crop:
    def __init__(self, crop_center, crop_window_size):
        self.crop_center      = crop_center
        self.crop_window_size = crop_window_size


    def __call__(self, img, center):
        crop_center      = self.crop_center
        crop_window_size = self.crop_window_size

        # ___/ IMG \___
        # Calcualte the crop window range...
        y_min = crop_center[0] - crop_window_size[0] // 2
        x_min = crop_center[1] - crop_window_size[1] // 2
        y_max = crop_center[0] + crop_window_size[0] // 2
        x_max = crop_center[1] + crop_window_size[1] // 2

        # Resolve over the bound issue...
        size_img_y, size_img_x = img.shape
        y_min = max(y_min, 0)
        x_min = max(x_min, 0)
        y_max = min(y_max, size_img_y)
        x_max = min(x_max, size_img_x)

        # Crop...
        img_crop = img[y_min:y_max, x_min:x_max]


        # ___/ CENTER \___
        # Obtain the original center...
        cy, cx = center

        # Calculate new center...
        cy_crop = cy - y_min
        cx_crop = cx - x_min
        center_crop = (cy_crop, cx_crop)

        return img_crop, center_crop




class Resize:
    def __init__(self, size_y, size_x):
        self.size_y = size_y
        self.size_x = size_x


    def __call__(self, img, center):
        size_y = self.size_y
        size_x = self.size_x

        img_resize = resize(img, (size_y, size_x), anti_aliasing = True)

        cy, cx = center
        size_img_y, size_img_x = img.shape
        cy_resize = cy * size_y / size_img_y
        cx_resize = cx * size_x / size_img_x
        center_resize = (cy_resize, cx_resize)

        return img_resize, center_resize




class RandomCenterCropZoom:
    def __init__(self, trim_factor_max = 0.2):
        self.trim_factor_max = trim_factor_max

        return None


    def __call__(self, img, center):
        trim_factor_max = self.trim_factor_max

        size_img_y, size_img_x = img.shape
        cy, cx = center

        # ___/ Trim \___
        trim_factor = np.random.uniform(low = 0, high = trim_factor_max)

        img_cy = size_img_y // 2
        img_cx = size_img_x // 2
        crop_window_size = (int(size_img_y * (1 - trim_factor)),
                            int(size_img_x * (1 - trim_factor)))
        cropper = Crop( crop_center = (img_cy, img_cx), crop_window_size = crop_window_size )

        img_crop, center_crop = cropper(img, center = center)

        # ___/ Zoom \___
        resizer = Resize(size_img_y, size_img_x)

        img_resize, center_resize = resizer(img_crop, center_crop)

        return img_resize, center_resize




class RandomCrop:
    def __init__(self, center_shift_max = (0, 0), crop_window_size = (200, 200)):
        self.center_shift_max = center_shift_max
        self.crop_window_size = crop_window_size

        return None


    def __call__(self, img, center):
        cy_shift_max, cx_shift_max = self.center_shift_max
        crop_window_size           = self.crop_window_size

        size_img_y, size_img_x = img.shape
        cy, cx = center

        cy_shift = np.random.randint(low = cy - cy_shift_max, high = cy + cy_shift_max)
        cx_shift = np.random.randint(low = cx - cx_shift_max, high = cx + cx_shift_max)

        crop = Crop( crop_center = (cy_shift, cx_shift), crop_window_size = crop_window_size )

        img_crop, center_crop = crop(img, center = center)

        return img_crop, center_crop




class RandomShift:
    def __init__(self, frac_y_shift_max = 0.01, frac_x_shift_max = 0.01):
        self.frac_y_shift_max = frac_y_shift_max
        self.frac_x_shift_max = frac_x_shift_max

    def __call__(self, img, center, verbose = False):
        frac_y_shift_max = self.frac_y_shift_max
        frac_x_shift_max = self.frac_x_shift_max

        # Get the size of the image...
        size_img_y, size_img_x = img.shape

        # Draw a random value for shifting along x and y, respectively...
        y_shift_abs_max = size_img_y * frac_y_shift_max
        y_shift_abs_max = y_shift_abs_max
        y_shift = random.uniform(-y_shift_abs_max, y_shift_abs_max)
        y_shift = int(y_shift)

        x_shift_abs_max = size_img_x * frac_x_shift_max
        x_shift_abs_max = int(x_shift_abs_max)
        x_shift = random.uniform(-x_shift_abs_max, x_shift_abs_max)
        x_shift = int(x_shift)

        # Determine the size of the super image...
        size_super_y = size_img_y + 2 * abs(y_shift)
        size_super_x = size_img_x + 2 * abs(x_shift)

        # Construct a super image by padding (with zero) the absolute y and x shift...
        super = np.zeros((size_super_y, size_super_x))

        # Move the image to the target area...
        target_y_min = abs(y_shift) + y_shift
        target_x_min = abs(x_shift) + x_shift
        target_y_max = size_img_y + target_y_min
        target_x_max = size_img_x + target_x_min
        super[target_y_min:target_y_max, target_x_min:target_x_max] = img[:]

        # Crop super...
        crop_y_min = abs(y_shift)
        crop_x_min = abs(x_shift)
        crop_y_max = size_img_y + crop_y_min
        crop_x_max = size_img_x + crop_x_min
        crop = super[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if verbose: print( f"y-shift = {y_shift}, x-shift = {x_shift}" )

        # Adjust center
        cy, cx = center
        cy_shift = cy + y_shift
        cx_shift = cx + x_shift
        center_shift = (cy_shift, cx_shift)

        return crop, center_shift




class RandomRotate:
    def __init__(self, angle_max = 360): 
        self.angle_max = angle_max

        return None

    def __call__(self, img, center):
        angle_max = self.angle_max

        angle = np.random.uniform(low = 0, high = angle_max)

        img_rot = rotate(img, angle = angle, center = center[::-1])    # Scikit image wants (x, y) instead of (y, x)

        return img_rot, center




class RandomPatch:
    """ Randomly place num_patch patch with the size of size_y * size_x onto an image.
    """

    def __init__(self, num_patch, size_patch_y,    size_patch_x, 
                                  var_patch_y = 0, var_patch_x = 0, 
                                  returns_mask = False):
        self.num_patch    = num_patch                   # ...Number of patches
        self.size_patch_y = size_patch_y                # ...Size of the patch in y dimension
        self.size_patch_x = size_patch_x                # ...Size of the patch in x dimension
        self.var_patch_y  = max(0, min(var_patch_y, 1)) # ...Percent variation with respect to the patch size in x dimension
        self.var_patch_x  = max(0, min(var_patch_x, 1)) # ...Percent variation with respect to the patch size in y dimension
        self.returns_mask = returns_mask                # ...Is it allowed to return a mask

        return None


    def __call__(self, img, center):
        # Get the size of the image...
        size_img_y, size_img_x = img.shape

        # Construct a mask of ones with the same size of the image...
        mask = np.ones_like(img)

        # Generate a number of random position...
        pos_y = np.random.randint(low = 0, high = size_img_y, size = self.num_patch)
        pos_x = np.random.randint(low = 0, high = size_img_x, size = self.num_patch)

        # Stack two column vectors to form an array of (x, y) indices...
        pos_y = pos_y.reshape(-1,1)
        pos_x = pos_x.reshape(-1,1)
        pos   = np.hstack((pos_y, pos_x))

        # Place patch of zeros at all pos as top-left corner...
        for (y, x) in pos:
            size_patch_y = self.size_patch_y
            size_patch_x = self.size_patch_x

            # Apply random variance...
            # Find the absolute max pixel to vary
            varsize_patch_y = int(size_patch_y * self.var_patch_y)
            varsize_patch_x = int(size_patch_x * self.var_patch_x)

            # Sample an integer from the min-max pixel to vary
            # Also allow user to set var_patch = 0 when variance is not desired
            delta_patch_y = np.random.randint(low = -varsize_patch_y, high = varsize_patch_y if varsize_patch_y else 1)
            delta_patch_x = np.random.randint(low = -varsize_patch_x, high = varsize_patch_x if varsize_patch_x else 1)

            # Apply the change
            size_patch_y += delta_patch_y
            size_patch_x += delta_patch_x

            # Find the limit of the bottom/right-end of the patch...
            y_end = min(y + size_patch_y, size_img_y)
            x_end = min(x + size_patch_x, size_img_x)

            # Patch the area with zeros...
            mask[y : y_end, x : x_end] = 0

        # Appy the mask...
        img_masked = mask * img

        # Construct the return value...
        # Parentheses are necessary
        output = img_masked if not self.returns_mask else (img_masked, mask)

        return output, center
