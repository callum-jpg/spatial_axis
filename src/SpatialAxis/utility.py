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
    image_shape: typing.Tuple[int, int]
) -> typing.Tuple[shapely.geometry.Polygon]:
    """Create 3 shapely box polygons, each containing the next smallest object.

    Args:
        image_shape (typing.Tuple[int, int]): Size of array to draw polygons in.

    Returns:
        typing.Tuple[shapely.geometry.Polygon]: Outer, middle and inner polygons.
    """

    minx, miny, maxx, maxy = 0, 0, image_shape[1], image_shape[0]

    outer = shapely.geometry.box(minx, miny, maxx, maxy)

    middle = shapely.affinity.scale(outer, 0.8, 0.8)

    inner = shapely.affinity.scale(outer, 0.4, 0.4)

    edge = shapely.difference(outer, middle)
    cortex = shapely.difference(middle, inner)
    medulla = inner

    shapes = [edge, cortex, medulla]

    return shapes
