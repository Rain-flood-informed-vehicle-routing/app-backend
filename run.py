import os
from app import create_app
import osmnx as ox
import networkx as nx

app = create_app()

if __name__ == '__main__':

    graph_filename = "sao_paulo.graphml"
    place_name = "São Paulo, Brazil"

    if not os.path.exists('./app/api/v1/modelo_previsao/graph.graphml'):
        graph = ox.graph_from_place(place_name, network_type='drive')
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        graph = ox.distance.add_edge_lengths(graph)

        ox.save_graphml(graph, './app/api/v1/modelo_previsao/graph.graphml') 


    app.run(host="0.0.0.0", debug=True, port=8080)