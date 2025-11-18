 python3 -m uvicorn app:app --reload --port 8000
SHAPEFILE_PATH=/Users/macbookpro/Desktop/algorithmiin shinjilgee ba zohiomj/biy daalt1/backend/data/mongolia-251021-free.shp/gis_osm_roads_free_1.shp
"""
    Чиглэлтэй граф (EPSG:3857 метр нэгж). NetworkX ашиглахгүй.
    nodes: np.ndarray shape=(N,2)  -> (x,y)
    edges: dict[u] = list[(v, weight, eid)]
    geom_by_edge: eid -> LineString (EPSG:3857)
    """