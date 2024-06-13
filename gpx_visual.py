import gpxpy
import gmplot
import matplotlib.pyplot as plt

# Чтение GPX файла
def read_gpx_file(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        return gpx.tracks[0].segments[0].points

# Визуализация маршрута на спутниковой карте
def visualize_route(points):
    # Получение координат точек маршрута
    lats = [point.latitude for point in points]
    lngs = [point.longitude for point in points]

    # Создание объекта карты спутникового типа
    gmap = gmplot.GoogleMapPlotter(lats[0], lngs[0], 14)
    gmap.map_type = "satellite"  # Установка типа карты на спутниковый

    # Добавление маркера на начальную точку
    gmap.marker(lats[0], lngs[0], 'cornflowerblue')

    # Добавление пути на карту
    gmap.plot(lats, lngs, 'cornflowerblue', edge_width=2)

    # Сохранение карты в HTML файл
    gmap.draw('route_map.html')

# Построение графика скорости для каждого километра маршрута
def plot_speed_graph(points):
    distances = []
    speeds = []

    # Расчет расстояния и скорости между точками
    for i in range(1, len(points)):
        distance = points[i].distance_2d(points[i-1]) / 1000  # перевод в километры
        speed = distance / (points[i].time - points[i-1].time).total_seconds() * 3600  # перевод в км/ч

        distances.append(distance)
        speeds.append(speed)

    # Кумулятивная сумма расстояний для километров
    cumulative_distances = [0]
    cumulative_speeds = [0]

    for i in range(len(distances)):
        cumulative_distances.append(cumulative_distances[-1] + distances[i])
        cumulative_speeds.append(speeds[i])

    # Расчет общей средней скорости
    total_distance = cumulative_distances[-1]
    total_time = points[-1].time - points[0].time
    average_speed = total_distance / total_time.total_seconds() * 3600  # перевод в км/ч

    # Построение графика скорости
    plt.plot(cumulative_distances, cumulative_speeds)
    plt.xlabel('Distance (km)')  # изменено на километры
    plt.ylabel('Speed (km/h)')  # изменено на километры в час
    plt.title('Speed vs Distance\nAverage Speed: {:.2f} km/h'.format(average_speed))  # добавление средней скорости в заголовок
    plt.grid(True)
    plt.savefig('speed_graph.png')
    plt.show()

# Путь к GPX файлу
gpx_file_path = 'your_track.gpx'

# Чтение GPX файла
points = read_gpx_file(gpx_file_path)

# Визуализация маршрута на карте
visualize_route(points)

# Построение графика скорости
plot_speed_graph(points)
