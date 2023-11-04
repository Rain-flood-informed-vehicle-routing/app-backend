import osmnx as ox
class Grafo:

    def __init__ (self, lat0, latoo, lon0, lonoo):
        """
        Initializes a graph object with latitude and longitude boundaries.
        Downloads and processes the graph data from OpenStreetMap within the
        specified boundaries.

        :param lat0 (float): Minimum latitude boundary.
        :param latoo (float): Maximum latitude boundary.
        :param lon0 (float): Minimum longitude boundary.
        :param lonoo (float): Maximum longitude boundary.
        """

        self.lat0 = lat0
        self.latoo = latoo
        self.lon0 = lon0
        self.lonoo = lonoo

        # Download and process the graph data within the specified boundaries
        self.graph = ox.graph_from_bbox(lat0, latoo, lon0, lonoo, network_type='drive')
        self.graph = ox.add_edge_speeds(self.graph)
        self.graph = ox.add_edge_travel_times(self.graph)
        self.graph = ox.distance.add_edge_lengths(self.graph)

    def get_nearest_node (self, lat, lon):
        """
        Finds the nearest node in the graph to the given latitude and
        longitude coordinates.

        :param lat (float): Latitude coordinate.
        :param lon (float): Longitude coordinate.

        :return node (int): Nearest node ID.
        """

        # Project the graph to enable us to use the nearest_nodes method
        graph_proj = ox.project_graph(self.graph)

        # Project the coordinates of the given point
        coords = [(lon, lat)]
        point = ox.projection.project_geometry(Point(coords))[0]
        x, y = point.x, point.y

        # Find the nearest node
        node = ox.distance.nearest_nodes(graph_proj, x, y, return_dist=False)

        return node

    def crop (self, initial_point, final_point):
        """
        Crops the graph based on the fastest path between two given nodes.

        :param initial_point (int): ID of the initial node.
        :param final_point (int): ID of the final node.

        :return cropped_graph (networkx.MultiDiGraph): Cropped graph
            containing only the nodes and edges around the fastest path.
        """

        # Find the fastest path between the initial and final nodes
        gmaps_path = nx.astar_path(self.graph, initial_point, final_point, weight='travel_time')

        # Retrieve latitude and longitude coordinates of the nodes along the
        # fastest path (aka Google Maps route)
        lats = [self.graph.nodes[node]['y'] for node in gmaps_path]
        lons = [self.graph.nodes[node]['x'] for node in gmaps_path]

        # Determine a tight bounding box around the path
        lat_min = min(lats)
        lat_max = max(lats)
        lon_min = min(lons)
        lon_max = max(lons)

        # Adjust the bounding box to ensure it contains the graph properly and
        # falls within the initial boundaries
        bbox_lat_min = max(self.lat0, lat_min - (lat_max - lat_min))
        bbox_lat_max = min(self.latoo, lat_max + (lat_max - lat_min))
        bbox_lon_min = max(self.lon0, lon_min - (lon_max - lon_min))
        bbox_lon_max = min(self.lonoo, lon_max + (lon_max - lon_min))

        # Crop the graph based on the adjusted bounding box
        self.graph = ox.truncate.truncate_graph_bbox(self.graph, bbox_lat_max, bbox_lat_min, bbox_lon_max, bbox_lon_min)

        return self.graph

    def reset (self):
        """
        Resets the graph to its original state, downloading and processing
        the graph data again.

        :return graph (networkx.MultiDiGraph): Reset graph.
        """
        self.graph = ox.graph_from_bbox(self.lat0, self.latoo, self.lon0, self.lonoo, network_type='drive')
        self.graph = ox.add_edge_speeds(self.graph)
        self.graph = ox.add_edge_travel_times(self.graph)
        self.graph = ox.distance.add_edge_lengths(self.graph)
        return self.graph

    def find_edge_by_nodes (self, node1, node2):
        """
        Finds an edge in the graph that links node1 to node2.

        :param node1 (int): Starting node ID.
        :param node2 (int): Ending node ID.

        :return None or edges[0] (None or tuple): Edge that links node1 to
            node2.
        """

        # Gather all the edges of the graph that start at node1 and end at node2,
        # or start at node2 and end at node1
        e1 = [e for e in self.graph.edges if e[0] == node1 and e[1] == node2]
        e2 = [e for e in self.graph.edges if e[0] == node2 and e[1] == node1]
        edges = [*e1, *e2]

        # If no edges were found, then print a message and return None.
        if len(edges) == 0:
            print(f"There are no edges linking {node1} to {node2} in graph.")
            return None

        # Otherwise, return the first edge that was found.
        else:
            return edges[0]
