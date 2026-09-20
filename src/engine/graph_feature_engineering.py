import json
import os
import pandas as pd
import networkx as nx

class GraphFeatureEngineer:
    """
    JSON formatındaki API çağrı graflarını okur ve topolojik özellikleri sayısallaştırır.
    İki farklı JSON formatını destekler:
      Format A: [{"_id": ..., "call_graph": [{"fromId": ..., "toId": ...}]}]
      Format B: {"graphs": [{"graph_id": ..., "nodes": [...], "edges": [{"source": ..., "target": ...}]}]}
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
        
        # Format tespiti
        if isinstance(data, list):
            # Format A: [{"_id": ..., "call_graph": [...]}]
            return self._process_format_a(data)
        elif isinstance(data, dict) and "graphs" in data:
            # Format B: {"graphs": [{"graph_id": ..., "edges": [...]}]}
            return self._process_format_b(data["graphs"])
        else:
            raise ValueError("Tanınmayan JSON graf formatı. 'call_graph' veya 'graphs' anahtarı bulunamadı.")
    
    def _process_format_a(self, data: list) -> pd.DataFrame:
        """Format A: call_graph + fromId/toId"""
        features_list = []
        
        for item in data:
            session_id = item.get("_id")
            call_graph = item.get("call_graph", [])
            
            G = nx.DiGraph()
            for edge in call_graph:
                from_node = edge.get("fromId")
                to_node = edge.get("toId")
                if from_node and to_node:
                    G.add_edge(from_node, to_node)
            
            features_list.append(self._extract_features(session_id, G))
            
        return pd.DataFrame(features_list)
    
    def _process_format_b(self, graphs: list) -> pd.DataFrame:
        """Format B: graphs/edges + source/target"""
        features_list = []
        
        for item in graphs:
            graph_id = item.get("graph_id")
            edges = item.get("edges", [])
            
            G = nx.DiGraph()
            # Önce node'ları ekle (varsa)
            for node in item.get("nodes", []):
                G.add_node(node.get("id"), **{k: v for k, v in node.items() if k != "id"})
            
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                if source and target:
                    G.add_edge(source, target)
            
            features_list.append(self._extract_features(graph_id, G))
            
        return pd.DataFrame(features_list)
    
    def _extract_features(self, item_id, G: nx.DiGraph) -> dict:
        """Bir graftan topolojik metrikleri çıkarır."""
        num_nodes = G.number_of_nodes()
        
        if num_nodes == 0:
            return {
                "_id": item_id,
                "graph_node_count": 0,
                "graph_edge_count": 0,
                "graph_density": 0.0,
                "graph_max_in_degree": 0,
                "graph_max_out_degree": 0,
                "graph_self_loops": 0
            }
        
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        in_degrees = dict(G.in_degree()).values()
        out_degrees = dict(G.out_degree()).values()
        
        max_in_degree = max(in_degrees) if in_degrees else 0
        max_out_degree = max(out_degrees) if out_degrees else 0
        
        self_loops = nx.number_of_selfloops(G)
        
        return {
            "_id": item_id,
            "graph_node_count": num_nodes,
            "graph_edge_count": num_edges,
            "graph_density": density,
            "graph_max_in_degree": max_in_degree,
            "graph_max_out_degree": max_out_degree,
            "graph_self_loops": self_loops
        }

