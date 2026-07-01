import json
import os
from typing import Any

class AMVDGExporter:
    """
    Handles the final serialization of the 2D primitives and topology mappings
    into the standard AMVDG JSON schema.
    """
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        
    def export(self, primitives: list[dict[str, Any]], metadata: dict[str, Any] = None):
        """
        Exports the enriched primitives to JSON.
        
        :param primitives: List of 2D primitives with `topo_origins` already injected.
        :param metadata: Optional metadata (e.g. camera orientation, object ID).
        """
        data = {
            "metadata": metadata or {},
            "primitives": []
        }
        
        for i, prim in enumerate(primitives):
            # Ensure an ID is set for schema compliance
            if "id" not in prim:
                prim["id"] = f"P_{i}"
                
            data["primitives"].append(prim)
            
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Exported AMVDG to {self.output_path}")

if __name__ == "__main__":
    # Simple test
    exporter = AMVDGExporter("/tmp/test_amvdg.json")
    exporter.export([
        {"type": "line", "p1": (0,0), "p2": (1,1), "topo_origins": ["Edge_0", "Face_1"]}
    ], metadata={"view": "front"})
