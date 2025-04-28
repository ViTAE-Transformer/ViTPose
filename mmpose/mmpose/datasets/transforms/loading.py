# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import LoadImageFromFile
import mmcv
import mmengine

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if "img" not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = super().transform(results)
            else:
                img = results["img"]
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if "img_path" not in results:
                    results["img_path"] = None
                results["img_shape"] = img.shape[:2]
                results["ori_shape"] = img.shape[:2]
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.' "Please check whether the file exists."
            )
            raise e

        return results


@TRANSFORMS.register_module()
class LoadImageFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self, to_float32=False, color_type="color", channel_order="rgb", file_client_args=dict(backend="disk")
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_image(self, path):
        img_bytes = self.file_client.get(path)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if img is None:
            raise ValueError(f"Fail to read {path}")
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmengine.FileClient(**self.file_client_args)

        image_file = results.get("img_path", None)

        if isinstance(image_file, (list, tuple)):
            # Load images from a list of paths
            results["img"] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            results["img"] = self._read_image(image_file)
        else:
            if "img" not in results:
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError("Either `image_file` or `img` should exist in " "results.")
            assert isinstance(results["img"], np.ndarray)
            if self.color_type == "color" and self.channel_order == "rgb":
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                if results["img"].ndim == 3:
                    results["img"] = mmcv.bgr2rgb(results["img"])
                elif results["img"].ndim == 4:
                    results["img"] = np.concatenate([mmcv.bgr2rgb(img) for img in results["img"]], axis=0)
                else:
                    raise ValueError('results["img"] has invalid shape ' f'{results["img"].shape}')

            results["image_file"] = None

        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"file_client_args={self.file_client_args})"
        )
        return repr_str
