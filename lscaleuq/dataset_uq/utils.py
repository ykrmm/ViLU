from PIL import Image


class ConvertImageToRGB:
    def __call__(self, image: Image) -> Image:
        return image.convert("RGB")
