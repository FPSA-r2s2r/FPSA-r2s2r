import open3d as o3d

in_path = "./data/objects/wrench/wrench_v3/wrench_attached_v3.obj"
out_path = "./data/objects/wrench/wrench_v3/wrench.obj"

mesh = o3d.io.read_triangle_mesh(in_path)

mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
mesh.remove_non_manifold_edges()
mesh.compute_vertex_normals()

print("is_edge_manifold allow_boundary_edges=True:", mesh.is_edge_manifold(allow_boundary_edges=True))
print("is_edge_manifold allow_boundary_edges=False:", mesh.is_edge_manifold(allow_boundary_edges=False))
print("is_vertex_manifold:", mesh.is_vertex_manifold())
print("is_watertight:", mesh.is_watertight())
print("is_orientable:", mesh.is_orientable())

o3d.io.write_triangle_mesh(out_path, mesh)
print("saved:", out_path)