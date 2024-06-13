segment = gpx.tracks[0].segments[0]
for ind, point in enumerate(segment.points):
    point.speed = segment.get_speed(ind)