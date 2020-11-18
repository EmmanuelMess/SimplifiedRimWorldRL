import shapely.geometry


def doesIntersect(shotStart: tuple, shotEnd: tuple, boxStart: tuple, width: float, height: float) -> bool:
    poly = shapely.geometry.Polygon([boxStart,
                                     (boxStart[0] + width, boxStart[1]),
                                     (boxStart[0] + width, boxStart[1] + height),
                                     (boxStart[0], boxStart[1] + height)])
    line = shapely.geometry.LineString([shotStart, shotEnd])

    return poly.intersects(line)

