import json
import os
import pandas as pd
import networkx as nx

class GraphFeatureEngineer:
    """
    JSON formatındaki API çağrı graflarını okur ve topolojik özellikleri sayısallaştırır.
    """
    
    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath
        
    def transform(self) -> pd.DataFrame:
        """
        JSON dosyasını okur, graf özelliklerini hesaplar ve özellikleri içeren DataFrame döndürür.
        """
        if not os.path.exists(self.json_filepath):
            raise FileNotFoundError(f"Graf dosyası bulunamadı: {self.json_filepath}")
            
        print(f"  [Graph Feature Engineer] Graf verisi işleniyor: {self.json_filepath} ...")
        with open(self.json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        features_list = []
        
        for item in data:
            session_id = item.get("_id")
            call_graph = item.get("call_graph", [])
            
            # Yönlü Graf (DiGraph) Oluşturma
            G = nx.DiGraph()
            
            # Kenarları (Edges) grafa ekle
            for edge in call_graph:
                from_node = edge.get("fromId")
                to_node = edge.get("toId")
                if from_node and to_node:
                    G.add_edge(from_node, to_node)
            
            # Eğer boş bir graf geldiyse hata almamak için sıfırla
            num_nodes = G.number_of_nodes()
            if num_nodes == 0:
                features_list.append({
                    "_id": session_id,
                    "graph_node_count": 0,
                    "graph_edge_count": 0,
                    "graph_density": 0.0,
                    "graph_max_in_degree": 0,
                    "graph_max_out_degree": 0,
                    "graph_self_loops": 0
                })
                continue
                
            # --- TOPOLOJİK METRİKLERİN HESAPLANMASI ---
            num_edges = G.number_of_edges()
            density = nx.density(G)
            
            # İç-Derece (In-Degree) ve Dış-Derece (Out-Degree) hesaplamaları
            in_degrees = dict(G.in_degree()).values()
            out_degrees = dict(G.out_degree()).values()
            
            max_in_degree = max(in_degrees) if in_degrees else 0
            max_out_degree = max(out_degrees) if out_degrees else 0
            
            # Kendine dönen düğümlerin (self-loops) sayısı
            self_loops = nx.number_of_selfloops(G)
            
            # Elde edilen metrikleri sözlüğe yaz
            features_list.append({
                "_id": session_id,
                "graph_node_count": num_nodes,
                "graph_edge_count": num_edges,
                "graph_density": density,
                "graph_max_in_degree": max_in_degree,
                "graph_max_out_degree": max_out_degree,
                "graph_self_loops": self_loops
            })
            
        return pd.DataFrame(features_list)
