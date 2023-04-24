# FGR+GICP Multiescale
ini = time.time()
pose_graph_FGR_GICP = full_registration_FGR_GICP(nuvens, voxel_size, k)
end = time.time()
print(f"Time taken by FGR+GICP-Multiescala: {end-ini} seconds")
time_FGR_GICP = end-ini


# Optimize pose-graphs with g2o (topological reduction to a circuit)
option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance = voxel_size/2, 
    edge_prune_threshold = 0.25, 
    reference_node = 0)
o3d.pipelines.registration.global_optimization(pose_graph_FGR_GICP, 
    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), 
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), 
    option)


# Import pose graphs
pose_graph_FGR = o3d.io.read_pose_graph("pose_graphs/pose_graph_FGR.json")
pose_graph_FGR_GICP = o3d.io.read_pose_graph("pose_graphs/pose_graph_FGR_GICP.json")


# Recover optimized absolute poses
poses_absolutas_FGR = [pose_graph_FGR.nodes[i].pose for i in range(n_nuvens)]
poses_absolutas_FGR_GICP = [pose_graph_FGR_GICP.nodes[i].pose for i in range(n_nuvens)]


# With the absolute poses recover the relative poses
poses_relativas_FGR = absolute_poses_to_relative_poses(poses_absolutas_FGR)
poses_relativas_FGR_GICP = absolute_poses_to_relative_poses(poses_absolutas_FGR_GICP)


# Add the loop-closure pose
T_loop_closure_FGR_GICP, _ = registro_FRG_GICP(nuvens[0], nuvens[-1], voxel_size)
poses_relativas_FGR_GICP.append(T_loop_closure_FGR_GICP.transformation)

