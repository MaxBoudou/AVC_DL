import cv2
import numpy as np

class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, X, Y):
        _X = cv2.resize(X, self.output_size)
        w, h = self.output_size
        c = Y.shape[-1]
        _Y = np.zeros((h, w, c))
        for i in range(Y.shape[-1]):
            _Y[..., i] = cv2.resize(Y[..., i], self.output_size)
        return _X, _Y

class Clip(object):
    def __init__(self, mini, maxi=None):
        if maxi is None:
            self.mini, self.maxi = 0, mini
        else:
            self.mini, self.maxi = mini, maxi

    def __call__(self, X, Y):
        mini_mask = np.where(X < self.mini)
        maxi_mask = np.where(X > self.maxi)
        X[mini_mask] = self.mini
        X[maxi_mask] = self.maxi
        return X, Y

class Normalize(object):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, X, Y):
        mini = np.min(X, self.axis)
        maxi = np.max(X, self.axis)
        X = (X - mini) / (maxi - mini)
        return X, Y

class Standardize(object):
    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, X, Y):
        mean =  np.mean(X, self.axis)
        std = np.std(X, self.axis)
        X = (X - mean) / std
        return X, Y

class Flip(object):
    def __call__(self, X, Y):
        for axis in [0, 1]:
            if np.random.rand(1) < 0.5:
                X = np.flip(X, axis)
                Y = np.flip(Y, axis)
        return X, Y

class Crop(object):
    def __init__(self, min_size_ratio, max_size_ratio=(1, 1)):
        self.min_size_ratio = np.array(list(min_size_ratio))
        self.max_size_ratio = np.array(list(max_size_ratio))

    def __call__(self, X, Y):
        size = np.array(X.shape[:2])
        mini = self.min_size_ratio * size
        maxi = self.max_size_ratio * size
        # random size
        h = np.random.randint(mini[0], maxi[0])
        w = np.random.randint(mini[1], maxi[1])
        # random place
        shift_h = np.random.randint(0, size[0] - h)
        shift_w = np.random.randint(0, size[1] - w)
        X = X[shift_h:shift_h+h, shift_w:shift_w+w]
        Y = Y[shift_h:shift_h+h, shift_w:shift_w+w]

        return X, Y

class Sharpen(object):
    def __init__(self, max_center=4):
        self.identity = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]])
        self.sharpen = np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]]) / 4

    def __call__(self, X, Y):

        sharp = self.sharpen * np.random.random() * self.max_center
        kernel = self.identity + sharp

        X = cv2.filter2D(X, -1, kernel)
        return X, Y

class GaussianBlur(object):
    def __init__(self, max_kernel=(7, 7)):
        if isinstance(max_kernel, int):
            max_kernel = (max_kernel, max_kernel)

        self.max_kernel = tuple((k + 1) // 2 for k in max_kernel)

    def __call__(self, X, Y):
        kernel_size = (
            np.random.randint(1, self.max_kernel[0]) * 2 + 1,
            np.random.randint(1, self.max_kernel[1]) * 2 + 1,
        )
        X = cv2.GaussianBlur(X, kernel_size, 0)
        return X, Y

class Perspective(object):
    def __init__(self,
                 max_ratio_translation=(0.2, 0.2, 0),
                 max_rotation=(10, 10, 360),
                 max_scale=(0.1, 0.1, 0.2),
                 max_shearing=(15, 15, 5)):

        self.max_ratio_translation = np.array(max_ratio_translation)
        self.max_rotation = np.array(max_rotation)
        self.max_scale = np.array(max_scale)
        self.max_shearing = np.array(max_shearing)

    def __call__(self, X, Y):

        # get the height and the width of the image
        h, w = X.shape[:2]
        max_translation = self.max_ratio_translation * np.array([w, h, 1])
        # get the values on each axis
        t_x, t_y, t_z = np.random.uniform(-1, 1, 3) * max_translation
        r_x, r_y, r_z = np.random.uniform(-1, 1, 3) * self.max_rotation
        sc_x, sc_y, sc_z = np.random.uniform(-1, 1, 3) * self.max_scale + 1
        sh_x, sh_y, sh_z = np.random.uniform(-1, 1, 3) * self.max_shearing

        # convert degree angles to rad
        theta_rx = np.deg2rad(r_x)
        theta_ry = np.deg2rad(r_y)
        theta_rz = np.deg2rad(r_z)
        theta_shx = np.deg2rad(sh_x)
        theta_shy = np.deg2rad(sh_y)
        theta_shz = np.deg2rad(sh_z)


        # compute its diagonal
        diag = (h ** 2 + w ** 2) ** 0.5
        # compute the focal length
        f = diag
        if np.sin(theta_rz) != 0:
            f /= 2 * np.sin(theta_rz)

        # set the image from cartesian to projective dimension
        H_M = np.array([[1, 0, -w / 2],
                        [0, 1, -h / 2],
                        [0, 0,      1],
                        [0, 0,      1]])
        # set the image projective to carrtesian dimension
        Hp_M = np.array([[f, 0, w / 2, 0],
                         [0, f, h / 2, 0],
                         [0, 0,     1, 0]])

        # adjust the translation on z
        t_z = (f - t_z) / sc_z ** 2
        # translation matrix to translate the image
        T_M = np.array([[1, 0, 0, t_x],
                        [0, 1, 0, t_y],
                        [0, 0, 1, t_z],
                        [0, 0, 0,  1]])

        # calculate cos and sin of angles
        sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
        sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
        sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
        # get the rotation matrix on x axis
        R_Mx = np.array([[1,      0,       0, 0],
                         [0, cos_rx, -sin_rx, 0],
                         [0, sin_rx,  cos_rx, 0],
                         [0,      0,       0, 1]])
        # get the rotation matrix on y axis
        R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                         [     0, 1,       0, 0],
                         [sin_ry, 0,  cos_ry, 0],
                         [     0, 0,       0, 1]])
        # get the rotation matrix on z axis
        R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                         [sin_rz,  cos_rz, 0, 0],
                         [     0,       0, 1, 0],
                         [     0,       0, 0, 1]])
        # compute the full rotation matrix
        R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

        # get the scaling matrix
        Sc_M = np.array([[sc_x,     0,    0, 0],
                         [   0,  sc_y,    0, 0],
                         [   0,     0, sc_z, 0],
                         [   0,     0,    0, 1]])

        # get the tan of angles
        tan_shx = np.tan(theta_shx)
        tan_shy = np.tan(theta_shy)
        tan_shz = np.tan(theta_shz)
        # get the shearing matrix on x axis
        Sh_Mx = np.array([[      1, 0, 0, 0],
                          [tan_shy, 1, 0, 0],
                          [tan_shz, 0, 1, 0],
                          [      0, 0, 0, 1]])
        # get the shearing matrix on y axis
        Sh_My = np.array([[1, tan_shx, 0, 0],
                          [0,       1, 0, 0],
                          [0, tan_shz, 1, 0],
                          [0,       0, 0, 1]])
        # get the shearing matrix on z axis
        Sh_Mz = np.array([[1, 0, tan_shx, 0],
                          [0, 1, tan_shy, 0],
                          [0, 0,       1, 0],
                          [0, 0,       0, 1]])
        # compute the full shearing matrix
        Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)

        Identity = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        # compute the full transform matrix
        M = Identity
        M = np.dot(Sh_M, M)
        M = np.dot(R_M,  M)
        M = np.dot(Sc_M, M)
        M = np.dot(T_M,  M)
        M = np.dot(Hp_M, np.dot(M, H_M))
        # apply the transformation
        X = cv2.warpPerspective(X, M, (w, h))
        Y = cv2.warpPerspective(Y, M, (w, h))
        return X, Y

class Cutout(object):
    def __init__(self,
                 min_size_ratio,
                 max_size_ratio,
                 channel_wise=False,
                 crop_target=True,
                 max_crop=10,
                 replacement=0):
        self.min_size_ratio = np.array(list(min_size_ratio))
        self.max_size_ratio = np.array(list(max_size_ratio))
        self.channel_wise = channel_wise
        self.crop_target = crop_target
        self.max_crop = max_crop
        self.replacement = replacement

    def __call__(self, X, Y):
        size = np.array(X.shape[:2])
        mini = self.min_size_ratio * size
        maxi = self.max_size_ratio * size
        for _ in range(self.max_crop):
            # random size
            h = np.random.randint(mini[0], maxi[0])
            w = np.random.randint(mini[1], maxi[1])
            # random place
            shift_h = np.random.randint(0, size[0] - h)
            shift_w = np.random.randint(0, size[1] - w)
            if self.channel_wise:
                c = np.random.randint(0, X.shape[-1])
                X[shift_h:shift_h+h, shift_w:shift_w+w, c] = self.replacement
                if self.crop_target:
                    Y[shift_h:shift_h+h, shift_w:shift_w+w] = self.replacement
            else:
                X[shift_h:shift_h+h, shift_w:shift_w+w] = self.replacement
                if self.crop_target:
                    Y[shift_h:shift_h+h, shift_w:shift_w+w] = self.replacement
        return X, Y

class Brightness(object):
    def __init__(self, range_brightness=(-50, 50)):
        self.range_brightness = range_brightness

    def __call__(self, X, Y):
        brightness = np.random.randint(*self.range_brightness)
        X = X + brightness
        return X, Y

class Contrast(object):
    def __init__(self, range_contrast=(-50, 50)):
        self.range_contrast = range_contrast

    def __call__(self, X, Y):
        contrast = np.random.randint(*self.range_contrast)
        X = X * (contrast / 127 + 1) - contrast
        return X, Y

class GaussianNoise(object):
    def __init__(self, center=0, std=50):
        self.center = center
        self.std = std

    def __call__(self, X, Y):
        noise = np.random.normal(self.center, self.std, X.shape)
        X = X + noise
        return X, Y

class Vignetting(object):
    def __init__(self,
                 ratio_min_dist=0.2,
                 range_vignette=(0.2, 0.8),
                 random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, X, Y):
        h, w = X.shape[:2]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w/2, w/2, w), np.linspace(-h/2, h/2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        vignette = np.tile(vignette[..., None], [1, 1, 3])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        X = X * (1 + sign * vignette)

        return X, Y

class LensDistortion(object):
    def __init__(self, d_coef=(0.15, 0.15, 0.1, 0.1, 0.05)):
        self.d_coef = np.array(d_coef)

    def __call__(self, X, Y):

        # get the height and the width of the image
        h, w = X.shape[:2]

        # compute its diagonal
        f = (h ** 2 + w ** 2) ** 0.5

        # set the image projective to carrtesian dimension
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0,     1]])

        d_coef = self.d_coef * np.random.random(5) # value
        d_coef = d_coef * (2 * (np.random.random(5) < 0.5) - 1) # sign
        # Generate new camera matrix from parameters
        M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)

        # Generate look-up tables for remapping the camera image
        remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

        # Remap the original image to a new image
        X = cv2.remap(X, *remap, cv2.INTER_LINEAR)
        Y = cv2.remap(Y, *remap, cv2.INTER_LINEAR)
        return X, Y

class ToTensor(object):
    def __init__(self, X_type=None, Y_type=None):
        # must bu torch types
        self.X_type = X_type
        self.Y_type = Y_type

    def __call__(self, X, Y):

        # swap color axis because
        # numpy img_shape: H x W x C
        # torch img_shape: C X H X W
        X = X.transpose((2, 0, 1))
        Y = Y.transpose((2, 0, 1))

        # convert to tensor
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        if self.X_type is not None:
            X = X.type(self.X_type)
        if self.Y_type is not None:
            Y = Y.type(self.Y_type)
        return X, Y

class ToType(object):
    def __init__(self, X_type, Y_type=None):
        # must bu numpy types
        self.X_type = X_type
        self.Y_type = Y_type if Y_type is not None else X_type

    def __call__(self, X, Y):
        return X.astype(self.X_type), Y.astype(self.Y_type)


# LIGHT_TRANSFORM = [
#     ToType(np.float),
#     Resize((960, 640)),
#     Flip(),
#     Brightness((-10, 10)),
#     Contrast((-10, 10)),
#     GaussianNoise(0, 5),
#     Clip(0, 255),
#     Normalize(),
# ]

# MEDIUM_TRANSFORM = [
#     ToType(np.float),
#     Normalize(),
#     Resize((960, 640)),
#     Flip(),
#     Brightness((-20, 20)),
#     Contrast((-20, 20)),
#     GaussianBlur(max_kernel=(5, 5)),
#     GaussianNoise(0, 15),
#     Vignetting(ratio_min_dist=0.1,
#                range_vignette=(0.1, 0.7),
#                random_sign=True),
#     LensDistortion(d_coef=(0.1, 0.1, 0.05, 0.05, 0.05)),
#     Perspective(max_ratio_translation=(0.1, 0.1, 0),
#                 max_rotation=(5, 5, 90),
#                 max_scale=(0.1, 0.1, 0.2),
#                 max_shearing=(5, 5, 2)),
#     Clip(0, 255),
#     Normalize(),
# ]

TEST_TRANSFORM = [
    ToType(np.float),
    # Flip(),
    # Normalize(),   
    Brightness((-20, 20)),
    Contrast((-20, 20)),
    GaussianBlur(max_kernel=(5, 5)),
    GaussianNoise(0, 15),
    # Vignetting(ratio_min_dist=0.1,
    #            range_vignette=(0.1, 0.7),
    #            random_sign=True),
    LensDistortion(d_coef=(0.2, 0.2, 0.1, 0.1, 0.1)),
    Perspective(max_ratio_translation=(0.0, 0.0, 0),
                max_rotation=(0, 0, 0),
                max_scale=(0.1, 0.1, 0.1),
                max_shearing=(0, 0, 0)),
    Clip(mini = 0, maxi = 255),
]


if __name__ == '__main__':

    # cv2.namedWindow('Transformation', cv2.WINDOW_NORMAL)

    
    X = np.load('test_image.npy')
    print(X.shape)
    # X = cv2.imread('./original_dataset/images/16.jpg')
    Y = cv2.imread('./original_dataset/masks/16.jpg')
    # Y = cv2.circle(Y.astype(np.uint8), (100, 150), 50, (0, 128, 255), 10)
    # Y = cv2.rectangle(Y, (10, 50), (100, 150), (128, 255, 0), -1)

    # sample_img = np.hstack((X, Y))

    # apply the list of transformations
    for tranform in TEST_TRANSFORM:
        X, Y = tranform(X, Y)

    print(X,Y)
    # tranformed_sample_img = np.hstack((X, Y))

    # show it : CHECK THE IMAGES SIZES TO STACK AND THE NORMALIZATION/STANDARDIZATION
    # img = np.vstack((sample_img / np.max(sample_img),
    #                  tranformed_sample_img / np.max(tranformed_sample_img)))
    # cv2.imshow('Transformation', img)
    # cv2.waitKey(0)
