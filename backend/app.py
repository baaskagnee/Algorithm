import os, time, math, json
from collections import deque
from typing import List, Tuple, Dict
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, mapping
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/macbookpro/Desktop/algorithmiin shinjilgee ba zohiomj/biy daalt1/backend/.env")

print("DEBUG:", os.getcwd())
print("DEBUG SHAPEFILE_PATH:", os.getenv("SHAPEFILE_PATH"))

SHAPEFILE_PATH = os.getenv("SHAPEFILE_PATH")
if not SHAPEFILE_PATH or not os.path.exists(SHAPEFILE_PATH):
    raise RuntimeError("SHAPEFILE_PATH тохируул (absolute path) — .env эсвэл орчинд!")

DEFAULT_METRIC = os.getenv("DEFAULT_METRIC", "distance")  # distance|time
SNAP_TOLERANCE_M = float(os.getenv("SNAP_TOLERANCE_M"))

DFS_MAX_PATHS = int(os.getenv("DFS_MAX_PATHS"))
DFS_MAX_HOPS = int(os.getenv("DFS_MAX_HOPS"))
DFS_MAX_DISTANCE_KM = float(os.getenv("DFS_MAX_DISTANCE_KM"))

app = FastAPI(title="UB Routing (BFS/DFS/Dijkstra)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

class RoadGraph:
    def __init__(self, metric: str = "distance"):
        assert metric in ("distance", "time")
        self.metric = metric
        self.nodes = np.zeros((0,2), dtype=float)
        self.edges: Dict[int, List[Tuple[int,float,int]]] = {}
        self.geom_by_edge: Dict[int, LineString] = {}
        self._node_index = {}

    @staticmethod
    def _key(x: float, y: float, q=0.01):
        return (round(x/q)*q, round(y/q)*q)

    def _dist_m(self, a, b) -> float:
        dx = a[0]-b[0]; dy = a[1]-b[1]; return float(math.hypot(dx,dy))

    def _weight(self, seg: LineString, fclass: str, surface: str, maxspeed) -> float:
        length = float(seg.length)
        if self.metric == "distance":
            penalty = {
                "gravel": 1.2, "ground": 1.3, "dirt": 1.35, "unpaved": 1.25, "sand": 1.5
            }.get((surface or "").lower(), 1.0)
            return length * penalty
        speeds = {"motorway":80,"trunk":70,"primary":60,"secondary":50,"tertiary":40,
                  "residential":30,"service":20,"unclassified":30}
        def parse_ms(v):
            if v is None: return None
            s=str(v).lower()
            num="".join(ch for ch in s if (ch.isdigit() or ch=="."))
            return float(num) if num else None
        sp = parse_ms(maxspeed) or speeds.get((fclass or "").lower(), 40.0)  # km/h
        sp_mps = sp*1000.0/3600.0
        sp_mps = max(sp_mps, 0.1)
        return length/sp_mps

    def load_shapefile(self, shp_path: str):
        gdf = gpd.read_file(shp_path)
        if str(gdf.crs).lower() != "epsg:3857":
            gdf = gdf.to_crs(epsg=3857)
        gdf = gdf[gdf.geometry.notnull()].explode(index_parts=False).reset_index(drop=True)

        nodes = []
        def get_id(x,y):
            k=self._key(x,y)
            if k in self._node_index: return self._node_index[k]
            nid=len(nodes); nodes.append([x,y]); self._node_index[k]=nid; return nid

        edges={}; geom_by_edge={}; eid=0

        for _,row in gdf.iterrows():
            geom=row.geometry
            if not isinstance(geom, LineString): continue
            coords=list(geom.coords)
            if len(coords)<2: continue

            fclass=(row.get("fclass") or "").lower()
            surface=row.get("surface")
            maxspeed=row.get("maxspeed")
            oneway = str(row.get("oneway") or "").lower() in {"yes","1","true"}
            access = (row.get("access") or "").lower()
            if access in {"no","private"}:
                continue

            for i in range(len(coords)-1):
                x1,y1=coords[i]; x2,y2=coords[i+1]
                u=get_id(x1,y1); v=get_id(x2,y2)
                seg=LineString([(x1,y1),(x2,y2)])
                w=self._weight(seg,fclass,surface,maxspeed)

                def add(a,b,w,sg):
                    nonlocal eid
                    edges.setdefault(a,[]).append((b,float(w),eid))
                    geom_by_edge[eid]=sg; eid+=1

                add(u,v,w,seg)
                if not oneway: add(v,u,w,seg)

        self.nodes=np.array(nodes,dtype=float)
        self.edges=edges
        self.geom_by_edge=geom_by_edge

    def nearest_node(self, x:float, y:float):
        if self.nodes.shape[0]==0: raise RuntimeError("Empty graph")
        dif=self.nodes - np.array([x,y])
        d2=np.einsum("ij,ij->i", dif, dif)
        idx=int(np.argmin(d2)); d=float(math.sqrt(d2[idx]))
        return idx,d

GRAPH = RoadGraph(metric=DEFAULT_METRIC)
GRAPH.load_shapefile(SHAPEFILE_PATH)

def dijkstra(s:int, t:int):
    dist={}; prev={}
    import heapq
    pq=[]; heapq.heappush(pq,(0.0,s))
    while pq:
        d,u=heapq.heappop(pq)
        if u in dist: continue
        dist[u]=d
        if u==t: break
        for v,w,_eid in GRAPH.edges.get(u,[]):
            if v in dist: continue
            nd=d+w; heapq.heappush(pq,(nd,v))
            if (v not in prev) or (nd < dist.get(v,1e100)): prev[v]=u
    if t not in dist: return None, float("inf")
    path=[t]; cur=t
    while cur!=s: cur=prev[cur]; path.append(cur)
    path.reverse(); return path, dist[t]

def bfs_fewest_hops(s:int, t:int, max_hops:int=10**9):
    q=deque([s]); prev={s:None}; hops={s:0}
    while q:
        u=q.popleft()
        if u==t: break
        for v,_w,_eid in GRAPH.edges.get(u,[]):
            if v not in prev:
                prev[v]=u; hops[v]=hops[u]+1
                if hops[v]<=max_hops: q.append(v)
    if t not in prev: return None, None
    path=[t]; cur=t
    while cur!=s: cur=prev[cur]; path.append(cur)
    path.reverse(); return path, hops[t]

def dfs_all_simple_paths(s: int, t: int, max_paths: int, max_hops: int, max_cost: float) -> List[Tuple[List[int], float]]:

    paths: List[Tuple[List[int], float]] = []

    def dfs(u: int, path: List[int], cost: float, visited: set):
        if len(paths) >= max_paths:
            return
        if u == t:
            paths.append((path.copy(), cost))
            return
        if (len(path) - 1) > max_hops or (cost > max_cost):
            return

        visited.add(u)
        for v, w, _eid in GRAPH.edges.get(u, []):
            if v not in visited:
                path.append(v)
                dfs(v, path, cost + w, visited)
                path.pop()
        visited.remove(u)

    dfs(s, [s], 0.0, set())
    return paths

def path_to_linestring(path_nodes: List[int]) -> LineString:
    coords=[tuple(GRAPH.nodes[n]) for n in path_nodes]
    return LineString(coords)

def feature(ls: LineString, props: dict):
    return {"type":"Feature","geometry":mapping(ls),"properties":props or {}}

def feature_collection(features: List[dict]):
    return {"type":"FeatureCollection","features":features}

def lonlat_to_3857(lon, lat):
    R=6378137.0
    x=lon*math.pi/180*R
    y=math.log(math.tan((90+lat)*math.pi/360))*R
    return x,y

class AllPathsRequest(BaseModel):
    start_lat: float; start_lng: float
    end_lat: float;   end_lng: float
    max_paths: int = DFS_MAX_PATHS
    max_hops: int = DFS_MAX_HOPS
    max_distance_km: float = DFS_MAX_DISTANCE_KM

@app.get("/health")
def health():
    return {
        "status":"ok",
        "nodes": int(GRAPH.nodes.shape[0]),
        "edges": sum(len(v) for v in GRAPH.edges.values()),
        "metric": GRAPH.metric,
        "snap_tolerance_m": SNAP_TOLERANCE_M
    }

def snap(lat,lng)->int:
    x,y=lonlat_to_3857(lng,lat)
    nid,d=GRAPH.nearest_node(x,y)
    if d>SNAP_TOLERANCE_M:
        raise HTTPException(400, detail=f"Too far from road network: {d:.1f} m")
    return nid

@app.get("/route/shortest")
def route_shortest(start_lat: float, start_lng: float, end_lat: float, end_lng: float):
    s=snap(start_lat,start_lng); t=snap(end_lat,end_lng)
    t0=time.perf_counter()
    path,cost=dijkstra(s,t)
    if not path: raise HTTPException(404,"No path found")
    ls=path_to_linestring(path)
    ms=(time.perf_counter()-t0)*1000
    return feature_collection([feature(ls,{"algorithm":"Dijkstra","cost":cost,"elapsed_ms":round(ms,1),"metric":GRAPH.metric})])

@app.get("/route/fewest_hops")
def route_fewest_hops(start_lat: float, start_lng: float, end_lat: float, end_lng: float):
    s=snap(start_lat,start_lng); t=snap(end_lat,end_lng)
    path,hops=bfs_fewest_hops(s,t)
    if not path: raise HTTPException(404,"No path (fewest hops) found")
    ls=path_to_linestring(path)
    return feature_collection([feature(ls,{"algorithm":"BFS","hops":int(hops)})])

@app.post("/route/all")
def route_all(req: AllPathsRequest):
    s=snap(req.start_lat, req.start_lng)
    t=snap(req.end_lat, req.end_lng)
    max_cost = req.max_distance_km*1000.0 if GRAPH.metric=="distance" else float("inf")
    paths = dfs_all_simple_paths(s, t, req.max_paths, req.max_hops, max_cost)
    if not paths:
        raise HTTPException(404,"No paths under limits")
    feats=[]
    for nodes,cost in paths:
        ls=path_to_linestring(nodes)
        feats.append(feature(ls,{"algorithm":"DFS","cost":cost,"hops":len(nodes)-1}))
    return feature_collection(feats)
