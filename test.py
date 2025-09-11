    #bbox = compute_bbox(gaussians.positions)
    ## 使用点云包围盒初始化体素边界，并设置体素分辨率
    #x_min, x_max, y_min, y_max, z_min, z_max, _ = bbox
    #bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    #resolution = (512, 512, 512)
    #opacoxels = Opacoxels(bounds=bounds, resolution=resolution, device="cuda" if torch.cuda.is_available() else "cpu")
#
#
    ## 写入不透明度到体素
    #positions_t = torch.from_numpy(gaussians.positions).float().to(opacoxels.device)
    #opacities_t = torch.from_numpy(gaussians.opacities).float().to(opacoxels.device)
    #opacoxels.update_from_points(positions_t, opacities_t)
#
    ## 采样一个点做 sanity check
    #center = torch.tensor([(x_min + x_max) * 0.5,
    #                       (y_min + y_max) * 0.5,
    #                       (z_min + z_max) * 0.5], device=opacoxels.device).float()
    #val = opacoxels.get_opacity_at_position(center)
    #print("Sampled center opacity:", float(val))
#
    ## 提取等值面网格并保存为 PLY
    #iso = 0.1
    #verts, faces, norms, _ = opacoxels.extract_surface(iso_level=iso)
#
    #os.makedirs(save_dir, exist_ok=True)
    #mesh_path = os.path.join(save_dir, "opacoxel_mesh.ply")


    ## 基于 positions 的泊松重建并保存为 PLY
    #if _O3D_OK:
    #    print("Running Poisson reconstruction from positions via Open3D...")
    #    pcd = o3d.geometry.PointCloud()
    #    pcd.points = o3d.utility.Vector3dVector(gaussians.positions.astype(np.float64))
# 
    #    # 法线估计
    #    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    #    pcd.orient_normals_consistent_tangent_plane(k=30)
# 
    #    # 泊松重建（depth 可调，越大越细）
    #    mesh_poisson, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
# 
    #    # 可选：使用包围盒裁剪网格到点云范围
    #    bbox_min = gaussians.positions.min(axis=0)
    #    bbox_max = gaussians.positions.max(axis=0)
    #    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
    #    mesh_poisson = mesh_poisson.crop(aabb)
    #    mesh_poisson.remove_degenerate_triangles()
    #    mesh_poisson.remove_duplicated_triangles()
    #    mesh_poisson.remove_duplicated_vertices()
    #    mesh_poisson.remove_non_manifold_edges()
# 
    #    poisson_path = os.path.join(save_dir, "poisson_mesh.ply")
    #    o3d.io.write_triangle_mesh(poisson_path, mesh_poisson)
    #    print(f"Saved Poisson mesh to {poisson_path} with {len(mesh_poisson.vertices)} vertices and {len(mesh_poisson.triangles)} faces.")
    #else:
    #    print("Open3D 未安装，跳过泊松重建。安装: pip install open3d")

    ## 快速方法：将点云体素化为占据体，再直接 Marching Cubes（更快）
    #print("Running fast voxel-based surface extraction from positions...")
    ## 使用更低分辨率加速（可按需调高），并将所有点 opacity 置 1
    #fast_res = (256, 256, 256)
    #opacoxels_fast = Opacoxels(bounds=bounds, resolution=fast_res, device=opacoxels.device)
    #ones_opacity = torch.ones((gaussians.positions.shape[0], 1), device=opacoxels_fast.device, dtype=torch.float32)
    #opacoxels_fast.update_from_points(torch.from_numpy(gaussians.positions).float().to(opacoxels_fast.device), ones_opacity)
    #fast_iso = 0.5
    #f_verts, f_faces, f_norms, _ = opacoxels_fast.extract_surface(iso_level=fast_iso)
# 
    #fast_mesh_path = os.path.join(save_dir, "fast_voxel_mesh.ply")
    #f_vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    #f_vertex_data = np.empty(f_verts.shape[0], dtype=f_vertex_dtype)
    #f_vertex_data["x"], f_vertex_data["y"], f_vertex_data["z"] = f_verts[:, 0], f_verts[:, 1], f_verts[:, 2]
    #f_vertex_data["nx"], f_vertex_data["ny"], f_vertex_data["nz"] = f_norms[:, 0], f_norms[:, 1], f_norms[:, 2]
    #f_face_dtype = [("vertex_indices", "i4", (3,))]
    #f_face_data = np.array([(f_faces[i].astype(np.int32),) for i in range(f_faces.shape[0])], dtype=f_face_dtype)
    #PlyData([PlyElement.describe(f_vertex_data, "vertex"), PlyElement.describe(f_face_data, "face")], text=True).write(fast_mesh_path)
    #print(f"Saved fast voxel mesh to {fast_mesh_path} with {f_verts.shape[0]} vertices and {f_faces.shape[0]} faces (iso={fast_iso}, res={fast_res}).")