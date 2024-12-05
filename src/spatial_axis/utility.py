import typing

import numpy
import shapely


def random_shapely_circles(
    image_shape: typing.Tuple[int, int],
    num_circles: int,
    min_radius: int = 15,
    max_radius: int = 35,
    seed: int = None,
) -> typing.List[shapely.geometry.Polygon]:
    """Create a series of shapely polygon circles with random
    sizes within min_radius and max_radius.

    Args:
        image_shape (typing.Tuple[int, int]): Shape of image that circles will be
        contained inside
        num_circles (int): Number of circles to draw.
        min_radius (int, optional): Minimum radius of drawn circles. Defaults to 15.
        max_radius (int, optional): Maximum radius of drawn circles. Defaults to 35.
        seed (int, optional): Random seed for drawing circles. Defaults to None.

    Returns:
        typing.List[shapely.geometry.Polygon]: List of shapely polygons.
    """
    circles = []

    numpy.random.seed(seed=seed)

    for _ in range(num_circles):
        x_center = numpy.random.uniform(0, image_shape[1])
        y_center = numpy.random.uniform(0, image_shape[0])
        radius = numpy.random.uniform(min_radius, max_radius)

        circle = shapely.geometry.Point(x_center, y_center).buffer(radius)

        circles.append(circle)

    return circles


def create_broad_annotation_polygons(
    image_shape: typing.Tuple[int, int],
    annotation_shape: typing.Literal["box", "circle"] = "box",
    num_levels: int = 3,
    downscale_factor: float = 0.6,
) -> typing.Tuple[shapely.geometry.Polygon]:
    """Create 3 shapely box polygons, each containing the next smallest object.

    Args:
        image_shape (typing.Tuple[int, int]): Size of array to draw polygons in.

    Returns:
        typing.Tuple[shapely.geometry.Polygon]: Outer, middle and inner polygons.
    """

    assert num_levels > 2, f"num_levels must be > 2, got {num_levels}"

    minx, miny, maxx, maxy = 0, 0, image_shape[1], image_shape[0]

    if annotation_shape.casefold() == "box":
        outer = shapely.geometry.box(minx, miny, maxx, maxy)
    elif annotation_shape.casefold() == "circle":
        print(max(maxx, maxy))
        outer = shapely.geometry.Point(maxx / 2, maxy / 2).buffer(max(maxx, maxy))

    shapes = [outer]

    for level in range(num_levels - 1):
        # Scale the previous polygon by the scale_factor
        scaled_shape = shapely.affinity.scale(
            shapes[level], downscale_factor, downscale_factor
        )
        # Find the difference of the original shape and the scaled shape
        prev_shape = shapely.difference(shapes[level], scaled_shape)
        # Add the difference shape to ith position
        shapes[level] = prev_shape
        # Add the scaled shape (no difference) to the i+1th position
        shapes.append(scaled_shape)

    return shapes
